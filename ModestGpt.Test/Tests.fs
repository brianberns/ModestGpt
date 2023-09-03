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

        do
            let encoder = Encoder.create 4 "aaabcbcaaabc"
            Assert.AreEqual(4, encoder.VocabularyMap.Count)   // a, b, c, bc
            Assert.AreEqual(1, encoder.Merges.Length)         // b + c -> bc

            let tokenKeys = Encoder.encode encoder "aabc"
            Assert.AreEqual(3, tokenKeys.Length)              // a, a, bc

        do
            let encoder = Encoder.create 5 "aaabcbcaaabc"
            Assert.AreEqual(5, encoder.VocabularyMap.Count)   // a, b, c, bc, aa
            Assert.AreEqual(2, encoder.Merges.Length)         // b + c -> bc, a + a -> aa

            let tokenKeys = Encoder.encode encoder "aabc"
            Assert.AreEqual(2, tokenKeys.Length)              // aa, bc
