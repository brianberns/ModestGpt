namespace ModestGpt

type Encoder =
    {
        VocabularyMap : Map<string, int>
        Merges : List<(string * string) * string>
    }

module Encoder =

    let create (maxVocabSize : int) (text : string) =

        let rec loop encoder (contents : string[]) =

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
                    let pairs =
                        seq {
                            yield! contentPairs
                            yield (Array.last contents, "")     // add pair at the end for the last element
                        }
                    (false, pairs)
                        ||> Seq.mapFold (fun merged (first', second') ->
                            if merged then
                                None, false                     // ignore this pair because previous pair was merged
                            elif (first', second') = (first, second) then
                                Some (first' + second'), true   // merge this pair
                            else
                                Some first', false)
                        |> fst
                        |> Seq.choose id
                        |> Seq.toArray

                loop encoder' contents'

            else encoder

        let encoder =
            {
                VocabularyMap =
                    Seq.indexed text
                        |> Seq.map (fun (i, c) ->
                            string c, i)
                        |> Map
                Merges = []
            }
        let contents =
            Seq.map string text
                |> Seq.toArray
        loop encoder contents

    let encode (encoder : Encoder) (text : string) =
        [| 0 |]

    let decode (encoder : Encoder) (encodedText : int[]) =
        "dummy"
