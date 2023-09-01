open TorchSharp
open type torch
open type TensorIndex
open FSharp.Core.Operators   // reclaim "float" and other F# operators

open ModestGpt

/// Dataset for the Sort problem. E.g. for problem length 6:
///    Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2
/// Which will feed into the transformer concatenated as:
///    Input:  0 0 2 1 0 1 0 0 0 1 1
///    Output: I I I I I 0 0 0 1 1 2
/// where I is -1, a special value that's ignored by the model
/// when calculating loss.
type SortDataset(count, ?length, ?numDigits) =
    inherit Dataset()

        // length of sequence to sort
    let length = defaultArg length 6

        // number of digits in vocabulary
    let numDigits = defaultArg numDigits 3

    let tensorPairs =
        Array.init count (fun _ ->

                // generate some random integers
                // e.g. "202010"
            let inp =
                torch.randint(
                    int64 numDigits,
                    size = [| int64 length |],
                    dtype = torch.long)
        
                // solve the task: i.e. sort
                // e.g. "000122"
            let sol = torch.sort(inp) |> fstv

                // concatenate the problem specification and the solution
                // e.g. "202010000122"
            let cat = torch.cat([|inp; sol|], dim = 0)

                // the inputs to the transformer will be the offset sequence
                // e.g. x: "20201000012"
                //      y:  "02010000122"
            let x = cat[Slice(stop = -1)].clone()
            let y = cat[Slice(1)].clone()

                // we only want to predict at output locations, mask out the loss at the input locations
                // e.g. "IIIII000122" where I is -1
            y[Slice(stop = length-1)] <- tensor -1
            x, y)

    /// Size of dataset.
    override _.Count with get() = tensorPairs.Length

    /// Get tensor pair by index.
    override _.GetTensor(idx) = tensorPairs[int idx]

    /// Vocabulary size. E.g. {0, 1, 2} -> 3 distinct values.
    member _.VocabSize = numDigits

    /// The length of the sequence that will feed into transformer, 
    /// containing concatenated input and the output, but -1 because
    /// the transformer starts making predictions at the last input
    /// element.
    member _.BlockSize = length * 2 - 1

module Program =

    ModestGpt.setSeed 0
    let dataset = new SortDataset(10000)

    let model =
        let config =
            {
                VocabSize = dataset.VocabSize
                BlockSize = dataset.BlockSize
                NumEmbed = 48
                NumLayer = 3
                NumHead = 3
                Dropout = 0.1
            }
        printfn $"Model config:\n{config}"
        new Gpt(config)

    let config =
        {
            Device = "cuda"
            MaxIters = Some 2000
            BatchSize = 64
            LearningRate = 5e-4
            Beta1 = 0.9
            Beta2 = 0.95
            WeightDecay = 0.1
            GradNormClip = 1.0
        }
    printfn $"Trainer config:\n{config}"
    printfn $"{ceil (float dataset.Count / float config.BatchSize)} batches/epoch"

    for progress in Trainer.run config model dataset do
        if progress.IterationNum % 100 = 0 then
            printfn "Iteration: %A, Epoch: %A, Duration: %.1f ms, Loss: %f"
                progress.IterationNum
                progress.EpochNum
                progress.Duration.TotalMilliseconds
                progress.Loss
