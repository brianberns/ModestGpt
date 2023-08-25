namespace ModestGpt

open System
open System.Collections.Generic

open TorchSharp
open TorchSharp.Modules
open type torch
open FSharp.Core.Operators   // reclaim "float" and other F# operators

type TrainerConfig =
    {
        Device : string
        NumWorkers : int
        MaxIters : int
        BatchSize : int
        LearningRate : float
        Beta1 : float
        Beta2 : float
        WeightDecay : float
        GradNormClip : float
    }

type Progress =
    {
        Device : string
        IterNum : int
        IterDt : TimeSpan
        Loss : float32
    }

module Trainer =

    let createOptimizer (model : nn.Module) config =

        // separate out all parameters to those that will and won't experience regularizing weight decay
        let parmGroups =
            [|
                for mdule in model.modules() do
                    match mdule :> obj with
                        | :? IWeightDecay as mdule ->
                            for parm, setting in mdule.ParameterSettings do
                                let wd =
                                    if setting then config.WeightDecay
                                    else 0.0
                                AdamW.ParamGroup(
                                    [ parm ],
                                    AdamW.Options(weight_decay = wd))
                        | _ -> ()
            |]
        assert(parmGroups.Length = 
            (model.named_parameters() |> Seq.length))
        torch.optim.AdamW(
            parmGroups,
            config.LearningRate,
            config.Beta1,
            config.Beta2)

    let run (config : TrainerConfig) (model : Gpt) dataset callback =

        // determine the device we'll train on
        let device =
            if config.Device = "auto" then
                if torch.cuda.is_available() then "cuda"
                else "cpu"
            else config.Device
        let model = model.``to``(device)
        do printfn $"running on device {device}"

        // setup the optimizer
        let optimizer = createOptimizer model config

        // setup the dataloader
        let train_loader =
            new DataLoader(dataset, config.BatchSize, shuffle=true, numWorker=config.NumWorkers)

        model.train()

        let rec loop iterNum iterTime (enumerator : IEnumerator<_>) =

            if enumerator.MoveNext() then

                let iterTime =
                    use _scope = torch.NewDisposeScope()

                    // fetch the next batch (x, y)
                    let (x : Tensor), (y : Tensor) = enumerator.Current
                    let x = x.``to``(device)
                    let y = y.``to``(device)

                    // forward the model
                    let _logits, loss = model.forward(x, y)

                    // backprop and update the parameters
                    optimizer.zero_grad((*set_to_none=true*))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GradNormClip) |> ignore
                    optimizer.step() |> ignore

                    let tnow = DateTime.Now
                    let iterDt = tnow - iterTime
                    callback {
                        Device = device
                        IterNum = iterNum
                        IterDt = iterDt
                        Loss = loss.item<float32>()
                    }
                    tnow

                // termination conditions
                if config.MaxIters <= 0 || iterNum < config.MaxIters then
                    loop (iterNum + 1) iterTime enumerator

            else
                train_loader.GetEnumerator() |> loop (iterNum + 1) iterTime

        train_loader.GetEnumerator() |> loop 0 DateTime.Now
