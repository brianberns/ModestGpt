open System

open TorchSharp
open type torch.TensorIndex
open FSharp.Core.Operators   // reclaim "float" and other F# operators

open ModestGpt
open TokenGpt

module Program =

        // prep for potential Unicode printing
    Console.OutputEncoding <- Text.Encoding.UTF8
    ModestGpt.setSeed 0   // this really isn't needed, but seems to supress a nasty TorchSharp warning

        // load model (this must match the original model configuration)
    let device = "cpu"
    let encoder = Encoder.load "Encoder.json"
    let config =
        {
            VocabSize = encoder.VocabSize
            BlockSize = 128
            NumEmbed = 512
            NumLayer = 8
            NumHead = 16
            Dropout = 0.1
        }
    let model = (new Gpt(config)).To(device)
    model.load("model.dat") |> ignore
    model.eval()

        // interactive "chat" loop
    let rec loop () =

        try
                // get next user input
            printfn ""
            printf "> "
            let context = Console.ReadLine()

                // determine completion
            let tokenKeys = Encoder.encode encoder context
            if tokenKeys.Length < config.BlockSize then
                let input =
                    torch.tensor(
                        tokenKeys,
                        device = device,
                        dtype = torch.long)[None, Ellipsis]
                let output =
                    model.Generate(
                        input,
                        config.BlockSize - tokenKeys.Length)[0]
                let completion =
                    output.data<int64>().ToArray()
                        |> Array.map int
                        |> Encoder.decode encoder
                printfn ""
                printfn "%s" completion

        with ex ->
            printfn ""
            printfn $"{ex.Message}"

        loop ()

    loop ()
