namespace ModestGpt

open System

type Encoder =
    {
        VocabularyMap : Map<string, int>
        Merges : List<(string * string) * string>
    }

module Encoder =

    /// Makes the given string printable.
    let private printable (str : string) =
        String [|
            for c in str do
                if Char.IsWhiteSpace(c) || Char.IsControl(c) then
                    yield! $"[{int c}]"
                else
                    yield c
        |]

    /// Initializes a non-merging encoder from the characters in
    /// the given text.
    let private initialize (text : string) =
        {
            VocabularyMap =
                set text
                    |> Seq.indexed
                    |> Seq.map (fun (i, c) ->
                        string c, i)
                    |> Map
            Merges = []
        }

    /// Makes a content array from the given text.
    let private toContents (text : string) =
        Seq.map string text
            |> Seq.toArray

    /// Merges occurrences of the given pair within the given pairs
    /// of content.
    let private merge contentPairs pair =
        printfn $"Merging tokens {Tuple2.map printable pair}"
        assert(
            contentPairs
                |> Seq.pairwise
                |> Seq.forall (fun ((_, a), (b, _)) -> a = b))
        let pairs =
            seq {
                yield! contentPairs
                yield (Array.last contentPairs |> snd, "")   // add pair at the end for the last element
            }
        (false, pairs)
            ||> Seq.mapFold (fun merged (first, second) ->
                if merged then
                    None, false                              // ignore this pair because previous pair was merged
                elif (first, second) = pair then
                    Some (first + second), true              // merge this pair
                else
                    Some first, false)
            |> fst
            |> Seq.choose id
            |> Seq.toArray

    /// Creates an encoder from the given text.
    let create maxVocabSize text =

        /// Attempts to add another token to the encoder.
        let rec loop encoder (contents : _[]) =

            if encoder.VocabularyMap.Count < maxVocabSize
                && contents.Length > 1 then

                    // find next pair of strings to merge into a token
                let contentPairs = Array.pairwise contents
                let first, second =
                    contentPairs
                        |> Seq.groupBy id
                        |> Seq.maxBy (snd >> Seq.length)
                        |> fst
                let token = first + second

                    // add the token to the encoder
                let encoder' =
                    {
                        VocabularyMap =
                            Map.add
                                token
                                encoder.VocabularyMap.Count
                                encoder.VocabularyMap
                        Merges =
                            let merge = (first, second), token
                            merge :: encoder.Merges
                    }

                    // merge occurrences of the token in the content
                let contents' =
                    merge contentPairs (first, second)

                loop encoder' contents'

            else encoder

        let encoder =
            loop (initialize text) (toContents text)
        { encoder with Merges = List.rev encoder.Merges }   // simpler merges first

    /// Encodes the given text.
    let encode encoder text =

        let mergeMap =
            encoder.Merges
                |> Seq.indexed
                |> Seq.map (fun (i, (pair, _)) -> pair, i)
                |> Map
        let tryFind pair =
            mergeMap
                |> Map.tryFind pair
                |> Option.defaultValue Int32.MaxValue

        /// Compresses the given text by repeatedly merging the most common
        /// pairs.
        let rec compress (contents : _[]) =

            if contents.Length > 1 then

                let contentPairs = Array.pairwise contents
                let first, second = Seq.minBy tryFind contentPairs

                if encoder.VocabularyMap.ContainsKey(first + second) then
                    merge contentPairs (first, second)
                        |> compress
                else
                    assert(tryFind (first, second) = Int32.MaxValue)
                    contents

            else contents

        toContents text
            |> compress
            |> Array.map (fun key ->
                encoder.VocabularyMap[key])

    /// Decodes the given encoded text.
    let decode (encoder : Encoder) (encodedText : int[]) =

        let decoder =
            encoder.VocabularyMap
                |> Seq.map (fun (KeyValue(key, value)) -> value, key)
                |> Map
        assert(decoder.Count = encoder.VocabularyMap.Count)

        encodedText
            |> Seq.map (fun key -> decoder[key])
            |> String.concat ""
