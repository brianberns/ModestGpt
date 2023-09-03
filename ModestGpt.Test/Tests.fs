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
            Assert.AreEqual(3, encoder.VocabularyMap["bc"])
            Assert.AreEqual(["b", "c", "bc"], encoder.Merges)

            let tokenKeys = Encoder.encode encoder "aabc"
            Assert.AreEqual([0; 0; 3], Seq.toList tokenKeys)  // a, a, bc

        do
            let encoder = Encoder.create 5 "aaabcbcaaabc"
            Assert.AreEqual(5, encoder.VocabularyMap.Count)   // a, b, c, bc, abc
            Assert.AreEqual(3, encoder.VocabularyMap["bc"])
            Assert.AreEqual(4, encoder.VocabularyMap["abc"])
            Assert.AreEqual(["b", "c", "bc"; "a", "bc", "abc"], encoder.Merges)

            let tokenKeys = Encoder.encode encoder "aabc"
            Assert.AreEqual([0; 4], Seq.toList tokenKeys)    // a, abc
