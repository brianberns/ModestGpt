namespace ModestGpt

open System.IO
open Microsoft.VisualStudio.TestTools.UnitTesting

[<TestClass>]
type EncoderTests() =

    let text = File.ReadAllText("Test.txt")
    let encoder, tokenKeys = Encoder.create 256 text

    member _.Text = text

    [<TestMethod>]
    member _.Symmetrical() =
        Assert.AreEqual(text, Encoder.decode encoder tokenKeys)
