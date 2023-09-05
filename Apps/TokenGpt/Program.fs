namespace TokenGpt

open System
open System.IO

open TorchSharp
open type torch.TensorIndex
open FSharp.Core.Operators   // reclaim "float" and other F# operators

open ModestGpt

type DatasetConfig =
    {
        InputFilePath : string
        BlockSize : int
        Context : string
    }

type TokenDataset(config) =
    inherit Dataset()

    let text =
        config.InputFilePath
            |> File.ReadLines
            |> Seq.map (fun line ->
                if line.StartsWith("<|endoftext|>") then "%%"
                else line)
            |> Seq.truncate 1000000
            |> String.concat "\n"
    do printfn $"Text length: {text.Length}"

    let encoder = Encoder.ofText text
    let tokenKeys = Encoder.encode encoder text
    do
        printfn $"Encoded length: {tokenKeys.Length}"
        Encoder.save "Encoder.json" encoder

    member _.Encode(str) = Encoder.encode encoder str

    member _.Decode(tokenKeys) = Encoder.decode encoder tokenKeys

    member _.VocabSize = encoder.VocabSize

    member _.BlockSize = config.BlockSize

    override _.Count with get() =
        assert(tokenKeys.Length > config.BlockSize)
        int64 (tokenKeys.Length - config.BlockSize)

    override _.GetTensor(idx) =

        let chunk = tokenKeys[int idx .. int idx + config.BlockSize]
        assert(chunk.Length = config.BlockSize + 1)

        torch.tensor(chunk[.. chunk.Length - 2], dtype = torch.long),
        torch.tensor(chunk[1 ..], dtype = torch.long)

module Program =

    Console.OutputEncoding <- Text.Encoding.UTF8
    ModestGpt.setSeed 0

    // construct the training dataset
    let datasetConfig =
        {
            InputFilePath = "Input.txt"
            BlockSize = 128
            Context = "Alice was very tired when she got back home, so she went"
        }
    let dataset = new TokenDataset(datasetConfig)

    let model =
        let modelConfig =
            {
                VocabSize = dataset.VocabSize
                BlockSize = dataset.BlockSize
                NumEmbed = 512
                NumLayer = 8
                NumHead = 16
                Dropout = 0.1
            }
        printfn $"Model config:\n{modelConfig}"
        new Gpt(modelConfig)

    let trainerConfig =
        {
            Device = "cuda"
            MaxIters = Option.None
            BatchSize = 36
            LearningRate = 5e-5
            Beta1 = 0.9
            Beta2 = 0.95
            WeightDecay = 0.1
            GradNormClip = 1.0
        }
    printfn $"Trainer config:\n{trainerConfig}"
    printfn $"{ceil (float dataset.Count / float trainerConfig.BatchSize)} batches/epoch"

    for progress in Trainer.run trainerConfig model dataset do

        if progress.IterationNum % 1000 = 0 then
            printfn "Iteration: %A, Epoch: %.5f, Duration: %.1f ms, Loss: %f"
                progress.IterationNum
                progress.EpochNum
                progress.Duration.TotalMilliseconds
                progress.Loss

        if progress.IterationNum % 10000 = 0 then
            model.eval()
            using (torch.no_grad()) (fun _ ->
                // sample from the model...
                let x =
                    torch.tensor(
                        dataset.Encode(datasetConfig.Context),
                        dtype = torch.long)
                let x = x[None, Ellipsis].To(trainerConfig.Device)
                let y = model.Generate(x, dataset.BlockSize, temperature = 1.0, sample = true, topK = 10)[0]
                let completion =
                    y.data<int64>().ToArray()
                        |> Array.map int
                        |> dataset.Decode
                printfn "%s" completion)
            model.save("model.dat") |> ignore
            // revert model to training mode
            model.train()
