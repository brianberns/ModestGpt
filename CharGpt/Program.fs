open System

open TorchSharp
open type torch.TensorIndex
open FSharp.Core.Operators   // reclaim "float" and other F# operators

open ModestGpt

type CharDatasetConfig =
    {
        block_size : int
    }

/// Emits batches of characters
type CharDataset(config, data : string) =
    inherit Dataset()

    let chars = set data
    let data_size, vocab_size_ = data.Length, chars.Count

    let stoi = Map [ for i, ch in Seq.indexed chars -> ch, i ]
    let itos = Map [ for i, ch in Seq.indexed chars -> i, ch ]

    do printfn "data has %d characters, %d unique." data_size vocab_size_

    static member get_default_config() =
        {
            block_size = 256
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
        let x = torch.tensor(dix[.. dix.Length-2], dtype = torch.long)
        let y = torch.tensor(dix[1 ..], dtype = torch.long)
        x, y

module Program =

    Console.OutputEncoding <- Text.Encoding.UTF8
    ModestGpt.setSeed 0

    // construct the training dataset
    let dataset =
        let text = System.IO.File.ReadAllText(@"Input.txt")
        new CharDataset({ block_size = 128 }, text)

    let model =
        new Gpt {
            NumLayer = 8
            NumHead = 16
            NumEmbed = 512
            VocabSize = dataset.get_vocab_size()
            BlockSize = dataset.get_block_size()
            Dropout = 0.1
        }

    // iteration callback
    let callback progress =

        if progress.IterationNum % 10 = 0 then
            printfn $"Iteration: {progress.IterationNum}, Duration: {progress.Duration.TotalMilliseconds:f1}ms, Loss: {progress.Loss}"

        if progress.IterationNum % 500 = 0 then
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

    let config =
        {
            Device = "cuda"
            NumWorkers = 4
            MaxIters = -1
            BatchSize = 64
            LearningRate = 5e-4
            Beta1 = 0.9
            Beta2 = 0.95
            WeightDecay = 0.1 // only applied on matmul weights
            GradNormClip = 1.0
        }

    Trainer.run config model dataset callback
