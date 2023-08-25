namespace ModestGpt

open System

open TorchSharp
open TorchSharp.Modules

#nowarn "40"   // allow recursive value

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
        IterationNum : int
        Duration : TimeSpan
        Loss : float32
    }

module Trainer =

    let createOptimizer (model : torch.nn.Module) config =

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
            match config.Device.ToLower(), torch.cuda.is_available() with
                | "cuda", true
                | "auto", true -> "cuda"
                | "cpu", _
                | "auto", false -> "cpu"
                | _ -> $"{config.Device} not supported"
        let model = model.To(device)
        do printfn $"running on device {device}"

        // setup the optimizer
        let optimizer = createOptimizer model config

        // setup the dataloader
        let tensorPairs =
            let loader =
                new DataLoader(dataset, config.BatchSize, shuffle=true, numWorker=config.NumWorkers)
            let rec pairs = seq { yield! loader; yield! pairs }
            if config.MaxIters < 0 then pairs
            else Seq.truncate (config.MaxIters + 1) pairs   // [0 .. MaxIters] inclusive

        model.train()

        ((DateTime.Now, 0), tensorPairs)
            ||> Seq.fold (fun (timeStart, iterNum) (input, target) ->
                use _scope = torch.NewDisposeScope()

                    // determine loss
                let loss =
                    let input = input.To(device)
                    let target = target.To(device)
                    model.GetLoss(input, target)

                let timeEnd = DateTime.Now
                callback {
                    Device = device
                    IterationNum = iterNum
                    Duration = timeEnd - timeStart
                    Loss = loss.item<float32>()
                }
                    // backprop and update the parameters
                optimizer.zero_grad((*set_to_none=true*))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(   // to-do: why?
                    model.parameters(),
                    config.GradNormClip) |> ignore
                optimizer.step() |> ignore

                timeEnd, iterNum + 1)

            |> ignore
