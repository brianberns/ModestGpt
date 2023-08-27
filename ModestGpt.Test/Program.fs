namespace ModestGpt

module Program =

    let text = EncoderTests().Text |> String.replicate 5
    let encoder = Encoder.create 512 text
    for key in encoder.VocabularyMap.Keys do
        printfn $"{Encoder.printable key}"
