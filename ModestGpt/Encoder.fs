namespace ModestGpt

open System

/// Byte-pair encoder (but not for bytes).
type Encoder =
    {
        /// Maps tokens to the their numeric representations.
        VocabularyMap : Map<string, int>

        /// Tokens to merge, in priority order. E.g. "do" + "nut" -> "donut".
        Merges : List<string * string * string>
    }

/// Character chategory.
type private Category =
    | Letter
    | Number
    | Punctuation
    | Whitespace
    | Symbol

module private Category =

    /// Determines the category of the given character.
    let ofChar c =
        if Char.IsLetter(c) || c = '\'' then Letter   // apostrophe is considered a letter
        elif Char.IsNumber(c) then Number
        elif Char.IsPunctuation(c) then Punctuation
        elif Char.IsWhiteSpace(c) || Char.IsControl(c) then Whitespace
        else Symbol

module Encoder =   // to-do: optimize this module for speed.

    /// Makes the given string printable.
    let printable (str : string) =
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
                assert(first <> "")
                if merged then
                    "", false                                // ignore this pair because previous pair was merged
                elif (first, second) = pair then
                    (first + second), true                   // merge this pair
                else
                    first, false)
            |> fst
            |> Seq.where ((<>) "")
            |> Seq.toArray

    /// Creates an encoder from the given text.
    let create maxVocabSize text =

        /// Attempts to add another token to the encoder.
        let rec loop encoder (contents : _[]) =

            if encoder.VocabularyMap.Count < maxVocabSize   // any more room?
                && contents.Length > 1 then                 // anything left to merge?

                    // find next pair of strings to merge into a token
                let contentPairs = Array.pairwise contents
                let first, second =
                    contentPairs
                        |> Seq.where (fun (first : string, second : string) ->
                            if second.Length > 1
                                && second[0] = ' '
                                && Category.ofChar second[1] = Letter then   // don't allow anything in front of a space-word
                                false
                            else
                                let catFirst = Category.ofChar first[0]
                                let catSecond = Category.ofChar second[0]
                                catFirst = catSecond
                                    || first = " " && catSecond = Letter)    // create space-word
                        |> Seq.groupBy id
                        |> Seq.maxBy (fun ((first, second), group) ->
                            Seq.length group, first.Length + second.Length)
                        |> fst
                let token = first + second
                printfn $"{encoder.VocabularyMap.Count}: Merging {printable first} + {printable second}"

                    // add the new token to the encoder
                let encoder' =
                    {
                        VocabularyMap =
                            Map.add
                                token
                                encoder.VocabularyMap.Count
                                encoder.VocabularyMap
                        Merges =
                            (first, second, token) :: encoder.Merges
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
                |> Seq.map (fun (i, (first, second, _)) ->
                    (first, second), i)
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
    let decode encoder encodedText =

        let decoder =
            encoder.VocabularyMap
                |> Map.toSeq
                |> Seq.map Tuple2.swap
                |> Map
        assert(decoder.Count = encoder.VocabularyMap.Count)

        encodedText
            |> Seq.map (fun tokenKey ->
                match Map.tryFind tokenKey decoder with
                    | Some token -> token
                    | _ -> failwith $"Unknown token: {tokenKey}")
            |> String.concat ""

    open System.IO
    open System.Text.Json

    /// Saves the given encoder to a file.
    let save path encoder =
        use stream = new FileStream(path, FileMode.Create)
        JsonSerializer.Serialize<Encoder>(stream, encoder)

    /// Loads an encoder from the given file.
    let load path =
        use stream = new FileStream(path, FileMode.Open)
        JsonSerializer.Deserialize<Encoder>(stream)
