namespace ModestGpt

open System

module Program =

    let text = EncoderTests().Text |> String.replicate 5
    let dtStart = DateTime.Now
    let encoder, tokenKeys = Encoder.create 512 text
    printfn $"{DateTime.Now - dtStart}"
