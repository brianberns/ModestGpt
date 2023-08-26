open System

open TorchSharp
open type torch.TensorIndex
open FSharp.Core.Operators   // reclaim "float" and other F# operators

open ModestGpt

type CharDatasetConfig =
    {
        BlockSize : int
    }

/// Emits batches of characters
type CharDataset(config, data : string) =
    inherit Dataset()

    let chars = set data
    let data_size, vocab_size_ = data.Length, chars.Count

    let stoi = Map [ for i, ch in Seq.indexed chars -> ch, i ]
    let itos = Map [ for i, ch in Seq.indexed chars -> i, ch ]

    do printfn "data has %d characters, %d unique." data_size vocab_size_

    member _.Itos(i) = itos[i]
    member _.Stoi(ch) = stoi[ch]

    member _.get_vocab_size() =
        vocab_size_

    member _.get_block_size() =
        config.BlockSize

    override _.Count with get() =
        int64 (data.Length - config.BlockSize)

    override _.GetTensor(idx) =
        // grab a chunk of (block_size + 1) characters from the data
        let chunk = data[int idx .. int idx + config.BlockSize]
        assert(chunk.Length = config.BlockSize + 1)
        // encode every character to an integer
        let dix = [| for ch in chunk -> stoi[ch] |]
        // return as tensors
        let x = torch.tensor(dix[.. dix.Length-2], dtype = torch.long)
        let y = torch.tensor(dix[1 ..], dtype = torch.long)
        x, y

module Program =

    Console.OutputEncoding <- Text.Encoding.UTF8
    ModestGpt.setSeed 0

    // construct the training dataset
    let dataset =
        let text = System.IO.File.ReadAllText(@"Input.txt")
        new CharDataset({ BlockSize = 256 }, text)

    let model =
        let config =
            {
                NumLayer = 6
                NumHead = 6
                NumEmbed = 192
                VocabSize = dataset.get_vocab_size()
                BlockSize = dataset.get_block_size()
                Dropout = 0.1
            }
        printfn $"Model config: {config}"
        new Gpt(config)

    let config =
        {
            Device = "cuda"
            MaxIters = -1
            BatchSize = 74
            LearningRate = 5e-4
            Beta1 = 0.9
            Beta2 = 0.95
            WeightDecay = 0.1 // only applied on matmul weights
            GradNormClip = 1.0
        }
    printfn $"Trainer config: {config}"
    printfn $"{ceil (float dataset.Count / float config.BatchSize)} batches/epoch"

    for progress in Trainer.run config model dataset do

        if progress.IterationNum % 100 = 0 then
            printfn "Iteration: %A, Epoch: %.5f, Duration: %.1f ms, Loss: %f"
                progress.IterationNum
                progress.EpochNum
                progress.Duration.TotalMilliseconds
                progress.Loss

        if progress.IterationNum % 1000 = 0 then
            model.eval()
            using (torch.no_grad()) (fun _ ->
                // sample from the model...
                let context = "It is "
                let x =
                    torch.tensor(
                        [| for ch in context -> dataset.Stoi(ch) |],
                        dtype = torch.long)
                let x = x[None, Ellipsis].To(progress.Device)
                let y = model.Generate(x, dataset.get_block_size(), temperature = 1.0, sample = true, topK = 10)[0]
                let completion = String ([| for i in y.data<int64>() -> dataset.Itos(int i) |])
                printfn "%s" completion)
            model.save("model.pt") |> ignore
            // revert model to training mode
            model.train()
