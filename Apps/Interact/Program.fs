open System

open TorchSharp
open type torch.TensorIndex
open FSharp.Core.Operators   // reclaim "float" and other F# operators

open ModestGpt

module Program =

    ModestGpt.setSeed 0

    let device = "cpu"
    let encoder = Encoder.load "Encoder.json"
    let config =
        {
            VocabSize = encoder.VocabularyMap.Count
            BlockSize = 128
            NumEmbed = 512
            NumLayer = 8
            NumHead = 16
            Dropout = 0.1
        }
    let model = (new Gpt(config)).To(device)
    model.load("model.dat") |> ignore

    let rec loop () =

        printfn ""
        printf "> "
        let context = Console.ReadLine()

        let x =
            torch.tensor(
                Encoder.encode encoder context,
                device = device,
                dtype = torch.long)[None, Ellipsis]
        let y =
            model.Generate(x, config.BlockSize)[0]
        let completion =
            y.data<int64>().ToArray()
                |> Array.map int
                |> Encoder.decode encoder
        printfn ""
        printfn "%s" completion

        loop ()

    loop ()
