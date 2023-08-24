namespace ModestGpt.Trainer

open System
open System.Collections.Generic

open TorchSharp
open TorchSharp.Modules
open type torch
open type TensorIndex
open FSharp.Core.Operators   // reclaim "float" and other F# operators

open ModestGpt
open ModestGpt.Model

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

module Trainer =

    let createOptimizer (model : Gpt) config =

        // separate out all parameters to those that will and won't experience regularizing weight decay
        let parmGroups =
            [|
                for mdule in model.modules() do
                    match mdule :> obj with
                        | :? IWeightDecay as module' ->
                            for parm, setting in module'.ParameterSettings do
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

    let run config model dataset =

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

                let iter_time =
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
                    let iter_dt = tnow - iterTime
                    if iterNum % 100 = 0 then
                        printfn $"iter {iterNum}: loss {loss.item<float32>()}"
                    tnow

                // termination conditions
                if config.MaxIters <= 0 || iterNum < config.MaxIters then
                    loop (iterNum + 1) iter_time enumerator

            else
                train_loader.GetEnumerator() |> loop (iterNum + 1) iterTime

        train_loader.GetEnumerator() |> loop 0 DateTime.Now

/// Dataset for the Sort problem. E.g. for problem length 6:
/// Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2
/// Which will feed into the transformer concatenated as:
/// input:  0 0 2 1 0 1 0 0 0 1 1
/// output: I I I I I 0 0 0 1 1 2
/// where I is "ignore", as the transformer is reading the input sequence
type SortDataset(split, ?length, ?num_digits) =
    inherit Dataset()

    let length = defaultArg length 6
    let num_digits = defaultArg num_digits 3

    do assert(List.contains split ["train"; "test"])

    let nTensorPairs = 10000

    let makeTensorPair _ =

        // use rejection sampling to generate an input example from the desired split
        let rec loop () =
            // generate some random integers
            let inp = torch.randint(int64 num_digits, size=[|int64 length|], dtype=torch.long)
            // half of the time let's try to boost the number of examples that 
            // have a large number of repeats, as this is what the model seems to struggle
            // with later in training, and they are kind of rare
            let reject =
                if torch.rand(1).item() < 0.5f then
                    let struct (unique, _, _) = inp.unique()
                    // too many unqiue digits, re-sample
                    unique.NumberOfElements > int64 length / 2L
                else false
            if reject then loop ()
            else
                // figure out if this generated example is train or test based on its hash
                let inp_split = if torch.rand(1).item() < 0.25f then "test" else "train" // designate 25% of examples as test
                if inp_split = split then
                    inp
                else loop ()

        let inp = loop ()

        /// First item of a value tuple.
        let fstv (struct (x, _)) = x

        /// Second item of a value tuple.
        let sndv (struct (_, y)) = y
        
        // solve the task: i.e. sort
        let sol = torch.sort(inp) |> fstv

        // concatenate the problem specification and the solution
        let cat = torch.cat([|inp; sol|], dim=0)

        // the inputs to the transformer will be the offset sequence
        let x = cat[Slice(stop = -1)].clone()
        let y = cat[Slice(1)].clone()
        // we only want to predict at output locations, mask out the loss at the input locations
        y[Slice(stop=length-1)] <- tensor -1
        x, y

    let tensorPairs =
        Array.init nTensorPairs makeTensorPair

    member _.Length = length

    override _.Count with get() = nTensorPairs

    member _.get_vocab_size() = num_digits

    member _.get_block_size() = length * 2 - 1

    override _.GetTensor(idx) =
        tensorPairs[int idx]

module Program =

    [<EntryPoint>]
    let main args =
        setSeed 0
        use model =
            new Gpt {
                NumLayer = 3
                NumHead = 3
                NumEmbed = 48
                VocabSize = 3
                BlockSize = 6 * 2 - 1
                Dropout = 0.1
            }
        let config =
            {
                Device = "auto"
                NumWorkers = 0
                MaxIters = 2000
                BatchSize = 64
                LearningRate = 5e-4
                Beta1 = 0.9
                Beta2 = 0.95
                WeightDecay = 0.1 // only applied on matmul weights
                GradNormClip = 1.0
            }
        use dataset = new SortDataset("train")
        Trainer.run config model dataset
                
        0
