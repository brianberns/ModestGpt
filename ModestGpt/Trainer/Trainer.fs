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
                    if iterNum % 10 = 0 then
                        printfn $"iter {iterNum}: loss {loss.item<float32>()}"
                    tnow

                // termination conditions
                if config.MaxIters <= 0 || iterNum < config.MaxIters then
                    loop (iterNum + 1) iter_time enumerator

            else
                train_loader.GetEnumerator() |> loop (iterNum + 1) iterTime

        train_loader.GetEnumerator() |> loop 0 DateTime.Now

type CharDatasetConfig =
    {
        block_size : int
    }

/// Emits batches of characters
type CharDataset(config, data : string) =
    inherit Dataset()

    let chars = set data
    let data_size, vocab_size_ = data.Length, chars.Count
    do printfn "data has %d characters, %d unique." data_size vocab_size_

    let stoi = Map [ for i, ch in Seq.indexed chars -> ch, i ]
    let itos = Map [ for i, ch in Seq.indexed chars -> i, ch ]

    static member get_default_config() =
        {
            block_size = 128
        }

    member _.Itos(i) = itos[i]
    member _.Stoi(ch) = stoi[ch]

    member _.get_vocab_size() =
        vocab_size_

    member _.get_block_size() =
        config.block_size

    override _.Count with get() =
        int64 (data.Length - config.block_size)

    override _.GetTensor(idx) =
        // grab a chunk of (block_size + 1) characters from the data
        let chunk = data[int idx .. int idx + config.block_size]
        assert(chunk.Length = config.block_size + 1)
        // encode every character to an integer
        let dix = [| for ch in chunk -> stoi[ch] |]
        // return as tensors
        let x = torch.tensor(dix[.. dix.Length-2], dtype=torch.long)
        let y = torch.tensor(dix[1 ..], dtype=torch.long)
        x, y

module Program =

    [<EntryPoint>]
    let main args =

        setSeed 0

        // construct the training dataset
        let dataset =
            let text = System.IO.File.ReadAllText(@"Trainer\Input.txt")
            new CharDataset({ block_size = 128 }, text)

        use model =
            new Gpt {
                NumLayer = 6
                NumHead = 6
                NumEmbed = 192
                VocabSize = dataset.get_vocab_size()
                BlockSize = dataset.get_block_size()
                Dropout = 0.1
            }

        let config =
            {
                Device = "auto"
                NumWorkers = 4
                MaxIters = -1
                BatchSize = 64
                LearningRate = 5e-4
                Beta1 = 0.9
                Beta2 = 0.95
                WeightDecay = 0.1 // only applied on matmul weights
                GradNormClip = 1.0
            }

        Trainer.run config model dataset
                
        0
