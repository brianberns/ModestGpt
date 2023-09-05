namespace TokenGpt

open System
open System.IO

open TorchSharp
open type torch.TensorIndex
open FSharp.Core.Operators   // reclaim "float" and other F# operators

open ModestGpt

/// Dataset configuration.
type DatasetConfig =
    {
        /// Path to input file.
        InputFilePath : string

        /// Maximum input sequence length (or "context" length).
        BlockSize : int

        /// Input context for evaluation.
        Context : string
    }

/// Dataset of text encoded into numeric tokens.
type TokenDataset(config) =
    inherit Dataset()

        // read training text
    let text =
        config.InputFilePath
            |> File.ReadLines
            |> Seq.map (fun line ->
                if line.StartsWith("<|endoftext|>") then "%%"   // avoid TikToken weirdness
                else line)
            |> Seq.truncate 1000000   // don't attempt to read a huge file entirely into memory.
            |> String.concat "\n"
    do printfn $"Text length: {text.Length}"

        // convert text to numeric tokens
    let encoder = Encoder.ofText text
    let tokenKeys = Encoder.encode encoder text
    do
        printfn $"Encoded length: {tokenKeys.Length}"
        Encoder.save "Encoder.json" encoder

    /// Encodes the given string as tokens.
    member _.Encode(str) = Encoder.encode encoder str

    /// Decodes the given tokens as a string.
    member _.Decode(tokenKeys) = Encoder.decode encoder tokenKeys

    /// Total number of tokens in the input text. Since this is typically
    /// far less than the number of tokens known to TikToken.
    member _.VocabSize = encoder.VocabSize

    /// Maxium input sequence length.
    member _.BlockSize = config.BlockSize

    /// Number of (input, output) tensor pairs in this dataset.
    override _.Count with get() =
        assert(tokenKeys.Length > config.BlockSize)
        int64 (tokenKeys.Length - config.BlockSize)

    /// Gets an (input, output) tensor pair.
    override _.GetTensor(idx) =

            // e.g. "31 41 59 26"
        let chunk = tokenKeys[int idx .. int idx + config.BlockSize]
        assert(chunk.Length = config.BlockSize + 1)

            // e.g. "31 41 59" and "41 59 26"
        let input = torch.tensor(chunk[.. chunk.Length - 2], dtype = torch.long)
        let output = torch.tensor(chunk[1 ..], dtype = torch.long)
        input, output

module Program =

        // prep for potential Unicode printing
    Console.OutputEncoding <- Text.Encoding.UTF8

        // run deterministically
    ModestGpt.setSeed 0

        // load the training dataset
    let datasetConfig =
        {
            InputFilePath = "Input.txt"
            BlockSize = 128
            Context = "Alice was very tired when she got back home, so she went"
        }
    let dataset = new TokenDataset(datasetConfig)

        // create the model
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

        // prepare to train
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

        // prepare example input for evaluation
    let input =
        torch.tensor(
            dataset.Encode(datasetConfig.Context),
            device = trainerConfig.Device,
            dtype = torch.long)[None, Ellipsis]

        // training loop
    for progress in Trainer.run trainerConfig model dataset do

            // display progress
        if progress.IterationNum % 1000 = 0 then
            printfn "Iteration: %A, Epoch: %.5f, Duration: %.1f ms, Loss: %f"
                progress.IterationNum
                progress.EpochNum
                progress.Duration.TotalMilliseconds
                progress.Loss

            // evaluate model
        if progress.IterationNum % 10000 = 0 then

                // switch to evaluation mode (e.g. no dropout)
            model.eval()
            using (torch.no_grad()) (fun _ ->

                    // sample from the model
                let output =
                    model.Generate(
                        input,
                        dataset.BlockSize,
                        temperature = 1.0,
                        sample = true,
                        topK = 10)[0]

                    // convert to text
                let completion =
                    output.data<int64>().ToArray()
                        |> Array.map int
                        |> dataset.Decode
                printfn "%s" completion)

                // save and revert model to training mode
            model.save("model.dat") |> ignore
            model.train()
