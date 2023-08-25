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

    ModestGpt.setSeed 0

    let model =
        new Gpt {
            NumLayer = 3
            NumHead = 3
            NumEmbed = 48
            VocabSize = 3
            BlockSize = 6 * 2 - 1
            Dropout = 0.1
        }

    let dataset = new SortDataset("train")

    // iteration callback
    let callback progress =

        if progress.IterationNum % 10 = 0 then
            printfn $"Iteration: {progress.IterationNum}, Duration: {progress.Duration.TotalMilliseconds:f1}ms, Loss: {progress.Loss}"

    let config =
        {
            Device = "cuda"
            NumWorkers = 0
            MaxIters = 2000
            BatchSize = 64
            LearningRate = 5e-4
            Beta1 = 0.9
            Beta2 = 0.95
            WeightDecay = 0.1 // only applied on matmul weights
            GradNormClip = 1.0
        }

    Trainer.run config model dataset callback
