open TorchSharp
open type torch
open type TensorIndex
open FSharp.Core.Operators   // reclaim "float" and other F# operators

open ModestGpt

/// Dataset for the Sort problem. E.g. for problem length 6:
/// Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2
/// Which will feed into the transformer concatenated as:
/// input:  0 0 2 1 0 1 0 0 0 1 1
/// output: I I I I I 0 0 0 1 1 2
/// where I is "ignore", as the transformer is reading the input sequence
type SortDataset(split, ?length, ?numDigits) =
    inherit Dataset()

    // length of sequence to sort
    let length = defaultArg length 6

    /// number of digits in vocabulary
    let numDigits = defaultArg numDigits 3

    do assert(List.contains split ["train"; "test"])

    let makeTensorPair _ =

        let rec loop () =

            // generate some random integers
            let inp =
                torch.randint(
                    int64 numDigits,
                    size = [| int64 length |],
                    dtype = torch.long)

            // figure out if this generated example is train or test based on its hash
            let inpSplit =
                if torch.rand(1).item() < 0.25f then "test"   // designate 25% of examples as test
                else "train"
            if inpSplit = split then inp
            else loop ()

        let inp = loop ()
        
        // solve the task: i.e. sort
        let sol = torch.sort(inp) |> fstv

        // concatenate the problem specification and the solution
        let cat = torch.cat([|inp; sol|], dim = 0)

        // the inputs to the transformer will be the offset sequence
        let x = cat[Slice(stop = -1)].clone()
        let y = cat[Slice(1)].clone()

        // we only want to predict at output locations, mask out the loss at the input locations
        y[Slice(stop = length-1)] <- tensor -1
        x, y

    let tensorPairs =
        Array.init 10000 makeTensorPair

    override _.Count with get() = tensorPairs.Length

    member _.VocabSize = numDigits

    member _.BlockSize = length * 2 - 1

    override _.GetTensor(idx) = tensorPairs[int idx]

module Program =

    ModestGpt.setSeed 0

    let dataset = new SortDataset("train")

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
        printfn $"Model config: {config}"
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
    printfn $"Trainer config: {config}"
    printfn $"{ceil (float dataset.Count / float config.BatchSize)} batches/epoch"

    for progress in Trainer.run config model dataset do
        if progress.IterationNum % 100 = 0 then
            printfn "Iteration: %A, Epoch: %A, Duration: %.1f ms, Loss: %f"
                progress.IterationNum
                progress.EpochNum
                progress.Duration.TotalMilliseconds
                progress.Loss
