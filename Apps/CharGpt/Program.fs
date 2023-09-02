open System
open System.IO

open TorchSharp
open type torch.TensorIndex
open FSharp.Core.Operators   // reclaim "float" and other F# operators

open ModestGpt

type CharDatasetConfig =
    {
        InputFilePath : string
        BlockSize : int
        Context : string
    }

/// Emits batches of characters
type CharDataset(config) =
    inherit Dataset()

    let text = File.ReadAllText(config.InputFilePath)
    let chars = set text
    let data_size, vocab_size_ = text.Length, chars.Count

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
        int64 (text.Length - config.BlockSize)

    override _.GetTensor(idx) =
        // grab a chunk of (block_size + 1) characters from the data
        let chunk = text[int idx .. int idx + config.BlockSize]
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
    let datasetConfig =
        {
            InputFilePath = "Input.txt"
            BlockSize = 256
            Context = "It is "
        }
    let dataset = new CharDataset(datasetConfig)

    let model =
        let modelConfig =
            {
                VocabSize = dataset.get_vocab_size()
                BlockSize = dataset.get_block_size()
                NumEmbed = 192
                NumLayer = 6
                NumHead = 6
                Dropout = 0.1
            }
        printfn $"Model config: {modelConfig}"
        new Gpt(modelConfig)

    let trainerConfig =
        {
            Device = "cuda"
            MaxIters = Option.None
            BatchSize = 74
            LearningRate = 5e-4
            Beta1 = 0.9
            Beta2 = 0.95
            WeightDecay = 0.1 // only applied on matmul weights
            GradNormClip = 1.0
        }
    printfn $"Trainer config: {trainerConfig}"
    printfn $"{ceil (float dataset.Count / float trainerConfig.BatchSize)} batches/epoch"

    for progress in Trainer.run trainerConfig model dataset do

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
                let x =
                    torch.tensor(
                        [| for ch in datasetConfig.Context -> dataset.Stoi(ch) |],
                        dtype = torch.long)
                let x = x[None, Ellipsis].To(trainerConfig.Device)
                let y = model.Generate(x, dataset.get_block_size(), temperature = 1.0, sample = true, topK = 10)[0]
                let completion = String ([| for i in y.data<int64>() -> dataset.Itos(int i) |])
                printfn "%s" completion)
            model.save("model.pt") |> ignore
            // revert model to training mode
            model.train()
