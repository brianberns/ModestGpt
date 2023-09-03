namespace ModestGpt

open System

/// Character category.
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

/// A list of tokens, indexed by the original location of their
/// first character in a text.
type private TokenList = Map<int, string>

module private TokenList =

    /// Creates a token list of individual characters.
    let create (text : string) : TokenList =
        text
            |> Seq.mapi (fun i c ->
                i, string c)
            |> Map

    /// Answers pairwise keys and values of the given token list.
    let pairwise (tokenList : TokenList) =
        tokenList
            |> Map.toSeq
            |> Seq.pairwise
            |> Seq.map (fun ((i, first), (j, second)) ->
                (i, j), (first, second))

/// Byte-pair encoder (but not for bytes).
type Encoder =
    {
        /// Maps tokens to the their numeric representations.
        VocabularyMap : Map<string, int>

        /// Tokens to merge, in priority order. E.g. "do" + "nut" -> "donut".
        Merges : List<string * string * string>
    }

module Encoder =

    /// Makes the given string printable.
    let printable (str : string) =
        String [|
            for c in str do
                if Char.IsWhiteSpace(c) || Char.IsControl(c) then
                    yield! $"[{int c}]"
                else
                    yield c
        |]

    /// Initializes a non-merging encoder from the characters in the
    /// given text.
    let private initialize (text : string) =
        {
            VocabularyMap =
                set text
                    |> Seq.mapi (fun i c ->
                        string c, i)
                    |> Map
            Merges = []
        }

    /// Merges occurrences of the given token within the given contents.
    let private merge indexPairs (token : string) (contents : TokenList) =
        (contents, indexPairs)
            ||> Seq.fold (fun acc (i, j) ->
                assert(j > i)
                assert(token.StartsWith(contents[i]))
                assert(token.EndsWith(contents[j]))
                acc
                    |> Map.add i token
                    |> Map.remove j)

    /// Creates an encoder from the given text.
    let create maxVocabSize text =

        /// Attempts to add another token to the encoder.
        let rec loop encoder contents =

            if encoder.VocabularyMap.Count < maxVocabSize then  // any more room?

                    // find next pair of strings to merge into a token
                TokenList.pairwise contents
                    |> Seq.where (fun (_, (first : string, second : string)) ->
                        if second.Length > 1
                            && second[0] = ' '
                            && Category.ofChar second[1] = Letter then               // don't allow anything in front of a space-word
                            false
                        else
                            let catFirst = Category.ofChar first[first.Length - 1]   // use last char in case of space-word
                            let catSecond = Category.ofChar second[0]
                            catFirst = catSecond
                                || first = " " && catSecond = Letter)                // create space-word
                    |> Seq.groupBy snd
                    |> Seq.tryMaxBy (fun ((first, second), group) ->
                        Seq.length group, first.Length + second.Length)
                    |> Option.map (fun ((first, second), group) ->

                            // add the new token to the encoder
                        let token = first + second
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
                        let indexPairs = Seq.map fst group
                        contents
                            |> merge indexPairs token
                            |> loop encoder')
                    |> Option.defaultValue encoder

            else encoder

        let encoder =
            loop (initialize text) (TokenList.create text)
        { encoder with Merges = List.rev encoder.Merges }   // simpler merges first

    /// Encodes the given text.
    let encode encoder text =

        let mergeMap =
            encoder.Merges
                |> Seq.mapi (fun i (first, second, _) ->
                    (first, second), i)
                |> Map
        let tryFind pair =
            mergeMap
                |> Map.tryFind pair
                |> Option.defaultValue Int32.MaxValue

        /// Compresses the given token list by repeatedly merging the most
        /// common pairs.
        let rec compress (contents : TokenList) =

            if contents.Count > 1 then

                let (first, second), indexPairs =
                    TokenList.pairwise contents
                        |> Seq.groupBy snd
                        |> Seq.map (fun (pair, group) ->
                            pair, Seq.map fst group)
                        |> Seq.minBy (fst >> tryFind)

                let token = first + second
                if encoder.VocabularyMap.ContainsKey(token) then
                    contents
                        |> merge indexPairs token
                        |> compress
                else
                    assert(tryFind (first, second) = Int32.MaxValue)
                    contents

            else contents

        TokenList.create text
            |> compress
            |> Map.values
            |> Seq.map (fun key ->
                encoder.VocabularyMap[key])
            |> Seq.toArray

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
