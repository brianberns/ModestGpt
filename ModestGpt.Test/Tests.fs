namespace ModestGpt

open System.IO
open Microsoft.VisualStudio.TestTools.UnitTesting

[<TestClass>]
type EncoderTests() =

    let text = File.ReadAllText("Test.txt")
    let encoder = Encoder.create 256 text
    let encode = Encoder.encode encoder
    let decode = Encoder.decode encoder

    member _.Text = text

    [<TestMethod>]
    member _.Symmetrical() =
        Assert.AreEqual(text, text |> encode |> decode)
