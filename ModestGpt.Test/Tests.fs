namespace ModestGpt

open System.IO
open Microsoft.VisualStudio.TestTools.UnitTesting

[<TestClass>]
type EncoderTests() =

    [<TestMethod>]
    member _.Symmetrical() =
        let text = File.ReadAllText("Test.txt")
        let encoder = Encoder.create 256 text
        Assert.AreEqual(text, text |> Encoder.encode encoder |> Encoder.decode encoder)

    [<TestMethod>]
    member _.Encode() =

        let encoder = Encoder.create 5 "aaabbbc"
        Assert.AreEqual(5, encoder.VocabularyMap.Count)   // a, b, c, aa, bb
        Assert.AreEqual(2, encoder.Merges.Length)         // a + a -> aa, b + b -> bb

        let tokenKeys = Encoder.encode encoder "ccccaabb"
        Assert.AreEqual(6, tokenKeys.Length)              // c, c, c, c, aa, bb
