open TorchSharp
open type torch.TensorIndex
open FSharp.Core.Operators   // reclaim "float" and other F# operators

open ModestGpt

module Program =

    let encoder = Encoder.load "Encoder.json"
    let config =
        {
            VocabSize = encoder.VocabularyMap.Count
            BlockSize = 192
            NumEmbed = 768
            NumLayer = 12
            NumHead = 12
            Dropout = 0.1
        }
    let model = new Gpt(config)
    model.load("model.dat").To("cuda") |> ignore

    let x =
        torch.tensor(
            Encoder.encode encoder "Hello ",
            device = "cuda",
            dtype = torch.long)[None, Ellipsis]
    let y = model.Generate(x, config.BlockSize, temperature = 1.0, sample = true, topK = 10)[0]
    let completion =
        y.data<int64>().ToArray()
            |> Array.map int
            |> Encoder.decode encoder
    printfn "%s" completion
