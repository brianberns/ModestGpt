open System
open System.IO

open TorchSharp
open type torch.TensorIndex
open FSharp.Core.Operators   // reclaim "float" and other F# operators

open ModestGpt

type DatasetConfig =
    {
        InputFilePath : string
        MaxVocabularySize : int
        BlockSize : int
        Context : string
    }

type TokenDataset(config) =
    inherit Dataset()

    let text = File.ReadAllText(config.InputFilePath)
    do printfn $"Text length: {text.Length}, {set text |> Set.count} distinct"

    let encoder =
        let path = "Encoder.json"
        if File.Exists(path) then
            Encoder.load path
        else
            printfn "Character frequencies:"
            text
                |> Seq.groupBy id
                |> Seq.map (fun (c, group) ->
                    c, Seq.length group)
                |> Seq.sortByDescending snd
                |> Seq.iter (fun (c, n) ->
                    printfn $"   {Encoder.printable (string c)}: {n}")
            let encoder = Encoder.create config.MaxVocabularySize text
            Encoder.save path encoder
            encoder
    do printfn $"Encoder vocabulary size: {encoder.VocabularyMap.Count}"

    let tokenKeys =
        let path = config.InputFilePath + ".bin"
        if File.Exists(path) then
            use stream = new FileStream(path, FileMode.Open)
            use reader = new BinaryReader(stream)
            let length = reader.ReadInt32()
            let tokenKeys =
                Array.init length (fun _ -> reader.ReadInt32())
            assert(Encoder.decode encoder tokenKeys = text)
            tokenKeys
        else
            let tokenKeys = Encoder.encode encoder text
            use stream = new FileStream(path, FileMode.Create)
            use writer = new BinaryWriter(stream)
            writer.Write(tokenKeys.Length)
            for tokenKey in tokenKeys do
                writer.Write(tokenKey)
            tokenKeys
    do printfn $"Encoded length: {tokenKeys.Length}"

    member _.Encoder = encoder

    member _.BlockSize = config.BlockSize

    override _.Count with get() =
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
            MaxVocabularySize = 1000
            BlockSize = 192
            Context = "It is"
        }
    let dataset = new TokenDataset(datasetConfig)

    let model =
        let modelConfig =
            {
                VocabSize = dataset.Encoder.VocabularyMap.Count
                BlockSize = dataset.BlockSize
                NumEmbed = 768
                NumLayer = 12
                NumHead = 12
                Dropout = 0.1
            }
        printfn $"Model config: {modelConfig}"
        new Gpt(modelConfig)

    let trainerConfig =
        {
            Device = "cuda"
            MaxIters = -1
            BatchSize = 14
            LearningRate = 3e-4
            Beta1 = 0.9
            Beta2 = 0.95
            WeightDecay = 0.1 // only applied on matmul weights
            GradNormClip = 1.0
        }
    printfn $"Trainer config: {trainerConfig}"
    printfn $"{ceil (float dataset.Count / float trainerConfig.BatchSize)} batches/epoch"

    for progress in Trainer.run trainerConfig model dataset do

        if progress.IterationNum % 500 = 0 then
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
                        Encoder.encode dataset.Encoder datasetConfig.Context,
                        dtype = torch.long)
                let x = x[None, Ellipsis].To(progress.Device)
                let y = model.Generate(x, dataset.BlockSize, temperature = 1.0, sample = true, topK = 10)[0]
                let completion =
                    y.data<int64>().ToArray()
                        |> Array.map int
                        |> Encoder.decode dataset.Encoder
                printfn "%s" completion)
            model.save("model.pt") |> ignore
            // revert model to training mode
            model.train()
