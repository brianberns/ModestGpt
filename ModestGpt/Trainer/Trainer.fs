namespace ModestGpt

open System

open TorchSharp
open TorchSharp.Modules

/// Trainer configuration.
type TrainerConfig =
    {
        /// Name of device to train on.
        Device : string

        /// Maximum number of training iterations.
        MaxIters : Option<int>

        /// Number of samples in a batch.
        BatchSize : int

        /// Learning rate.
        LearningRate : float

        /// Optimizer parameter.
        Beta1 : float

        /// Optimizer parameter.
        Beta2 : float

        /// Weight decay optimizer parameter.
        WeightDecay : float

        /// Gradient norm clip.
        GradNormClip : float
    }

type Progress =
    {
        /// Fractional epoch number. E.g. 1.00 means entire dataset has been processed once.
        EpochNum : float

        /// Iteration number, starting with 0.
        IterationNum : int

        /// Time elapsed during this iteration.
        Duration : TimeSpan

        /// Loss calculated during this iteration.
        Loss : float32
    }

module Trainer =

    /// Creates an optimizer for the given model.
    let private createOptimizer (model : torch.nn.Module) config =

            // determine which parameters will experience regularizing weight decay
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

    /// Runs a training loop for the given model using the given data.
    let run (config : TrainerConfig) (model : Gpt) dataset =

            // determine the device we'll train on
        let device = config.Device
        let model = model.To(device)
        do printfn $"Training on device {device}"

            // setup the optimizer
        let optimizer = createOptimizer model config

            // setup the data loader
        let tensorTuples =

            let tuples =
                let loader =
                    new DataLoader(
                        dataset,
                        config.BatchSize,
                        shuffle = true,
                        device = device)
                let rec loop epochNum =
                    seq {
                        for epochFrac, x, y in loader.Indexed do
                            yield float epochNum + epochFrac, x, y
                        yield! loop (epochNum + 1)
                    }
                loop 0

            config.MaxIters
                |> Option.map (fun maxIters ->
                    Seq.truncate (maxIters + 1) tuples)   // [0 .. MaxIters] inclusive
                |> Option.defaultValue tuples

            // put model into training mode
        model.train()

            // training loop
        ((DateTime.Now, 0, None), tensorTuples)
            ||> Seq.scan (fun (timeStart, iterNum, _) (epochNum, input, target) ->   // would prefer Seq.mapFold, but it is eager (for some reason)
                use _scope = torch.NewDisposeScope()

                    // determine loss
                let loss = model.GetLoss(input, target)

                    // backprop and update the parameters
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(   // to-do: why?
                    model.parameters(),
                    config.GradNormClip) |> ignore
                optimizer.step() |> ignore

                    // report progress
                let timeEnd = DateTime.Now
                let progress =
                    {
                        EpochNum = epochNum
                        IterationNum = iterNum
                        Duration = timeEnd - timeStart
                        Loss = loss.item<float32>()
                    }
                timeEnd, iterNum + 1, Some progress)

                // create stream of progress reports
            |> Seq.choose (fun (_, _, progressOpt) ->
                progressOpt)
