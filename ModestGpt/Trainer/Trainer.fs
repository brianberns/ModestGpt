﻿namespace ModestGpt

open System

open TorchSharp
open TorchSharp.Modules

type TrainerConfig =
    {
        Device : string
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
        EpochNum : int
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

    let run (config : TrainerConfig) (model : Gpt) dataset =

        // determine the device we'll train on
        let device = config.Device
        let model = model.To(device)
        do printfn $"running on device {device}"

        // setup the optimizer
        let optimizer = createOptimizer model config

        // setup the dataloader
        let tensorTuples =

            let tuples =
                let loader =
                    new DataLoader(
                        dataset,
                        config.BatchSize,
                        shuffle = true)
                let rec loop epochNum =
                    seq {
                        for x, y in loader do
                            yield epochNum, x, y
                        yield! loop (epochNum + 1)
                    }
                loop 0

            if config.MaxIters < 0 then tuples
            else Seq.truncate (config.MaxIters + 1) tuples   // [0 .. MaxIters] inclusive

        model.train()

        ((DateTime.Now, 0, None), tensorTuples)
            ||> Seq.scan (fun (timeStart, iterNum, _) (epochNum, input, target) ->   // would prefer Seq.mapFold, but it is eager (for some reason)
                use _scope = torch.NewDisposeScope()

                    // determine loss
                let loss =
                    let input = input.To(device)
                    let target = target.To(device)
                    model.GetLoss(input, target)

                    // backprop and update the parameters
                optimizer.zero_grad((*set_to_none=true*))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(   // to-do: why?
                    model.parameters(),
                    config.GradNormClip) |> ignore
                optimizer.step() |> ignore

                let timeEnd = DateTime.Now
                let progress =
                    {
                        Device = device
                        EpochNum = epochNum
                        IterationNum = iterNum
                        Duration = timeEnd - timeStart
                        Loss = loss.item<float32>()
                    }
                timeEnd, iterNum + 1, Some progress)

            |> Seq.choose (fun (_, _, progressOpt) ->
                progressOpt)
