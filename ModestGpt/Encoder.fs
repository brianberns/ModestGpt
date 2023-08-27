namespace ModestGpt

open System

type Encoder =
    {
        VocabularyMap : Map<string, int>
        Merges : List<(string * string) * string>
    }

module Encoder =

    let private printable (str : string) =
        String [|
            for c in str do
                if Char.IsAsciiLetterOrDigit(c) || Char.IsPunctuation(c) then
                    yield c
                else
                    yield! $"[{int c}]"
        |]

    /// Initializes an non-merging encoder with the characters
    /// in the given text.
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

    let private toContents (text : string) =
        Seq.map string text
            |> Seq.toArray

    /// Merges occurrences of the given pair with the given pairs
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

    let create maxVocabSize text =

        let rec loop encoder (contents : _[]) =

            if encoder.VocabularyMap.Count < maxVocabSize
                && contents.Length > 1 then

                let contentPairs = Array.pairwise contents
                let first, second =
                    contentPairs
                        |> Seq.groupBy id
                        |> Seq.maxBy (snd >> Seq.length)
                        |> fst
                let token = first + second

                let encoder' =
                    {
                        VocabularyMap =
                            Map.add
                                token
                                encoder.VocabularyMap.Count
                                encoder.VocabularyMap
                        Merges =
                            ((first, second), token) :: encoder.Merges
                    }
                let contents' =
                    merge contentPairs (first, second)

                loop encoder' contents'

            else encoder

        let encoder =
            loop (initialize text) (toContents text)
        { encoder with Merges = List.rev encoder.Merges }

    let encode encoder text =

        let tryFind pair =
            encoder.Merges
                |> List.tryFindIndex (fst >> (=) pair)

        let rec compress (contents : _[]) =

            if contents.Length > 1 then

                let contentPairs = Array.pairwise contents
                let first, second =
                    contentPairs
                        |> Seq.minBy (
                            tryFind
                                >> Option.defaultValue Int32.MaxValue)

                if encoder.VocabularyMap.ContainsKey(first + second) then
                    merge contentPairs (first, second)
                        |> compress
                else
                    assert(tryFind (first, second) |> Option.isNone)
                    contents

            else contents

        toContents text
            |> compress
            |> Array.map (fun key ->
                encoder.VocabularyMap[key])

    let decode (encoder : Encoder) (encodedText : int[]) =

        let decoder =
            encoder.VocabularyMap
                |> Seq.map (fun (KeyValue(key, value)) -> value, key)
                |> Map
        assert(decoder.Count = encoder.VocabularyMap.Count)

        encodedText
            |> Seq.map (fun key -> decoder[key])
            |> String.concat ""
