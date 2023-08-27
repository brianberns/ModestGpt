namespace ModestGpt

module Program =

    let text = EncoderTests().Text |> String.replicate 100
    let encoder = Encoder.create 256 text
    printfn "%A" <| Encoder.encode encoder text
