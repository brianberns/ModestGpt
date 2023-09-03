namespace TokenGpt

open System.IO
open Tiktoken
open ModestGpt

/// Tiktoken wrapper for small vocabularies.
type Encoder =
    {
        /// Tiktoken encoding name.
        EncodingName : string

        /// Tiktoken encoding.
        Encoding : Encoding

        /// Tiktoken keys.
        SmallToBigMap : int[]

        /// Map from local keys to Tiktoken keys.
        BigToSmallMap : Map<int, int>
    }

    /// Number of distinct tokens in this encoder.
    member this.VocabSize =
        assert(this.BigToSmallMap.Count = this.SmallToBigMap.Length)
        this.SmallToBigMap.Length

module Encoder =

    /// Creates an encoder.
    let private create encodingName encoding (bigKeys : Set<_>) =
        {
            EncodingName = encodingName
            Encoding = encoding
            SmallToBigMap =
                Seq.toArray bigKeys
            BigToSmallMap =
                bigKeys
                    |> Seq.mapi (fun smallKey bigKey ->
                        bigKey, smallKey)
                    |> Map
        }

    /// Creates an encoder for the given text.
    let ofText text =
        let encodingName = "cl100k_base"
        let encoding = Encoding.Get(encodingName)
        let bigKeys = encoding.Encode(text) |> set
        create encodingName encoding bigKeys

    /// Encodes the given text using the given encoder.
    let encode encoder text =
        encoder.Encoding
            .EncodeWithAllAllowedSpecial(text)
            |> Seq.map (fun bigKey ->
                match Map.tryFind bigKey encoder.BigToSmallMap with
                    | Some smallKey -> smallKey
                    | None ->
                        failwith $"Missing token {bigKey}: '{encoder.Encoding.Decode([bigKey])}'")
            |> Seq.toArray

    /// Decodes the given token keys to text.
    let decode encoder smallKeys =
        smallKeys
            |> Seq.map (Array.get encoder.SmallToBigMap)
            |> Seq.toArray
            |> encoder.Encoding.Decode

    open System.Text.Json

    /// Serialization type. Must be public.
    [<RequireQualifiedAccess>]
    type Json =
        {
            /// Tiktoken encoding name.
            EncodingName : string

            /// Tiktoken keys.
            Keys : int[]
        }

    /// Saves the given encoder to a file.
    let save path encoder =
        use stream = new FileStream(path, FileMode.Create)
        let value =
            {
                Json.EncodingName = encoder.EncodingName
                Json.Keys = encoder.SmallToBigMap
            }
        JsonSerializer.Serialize(stream, value)

    /// Loads an encoder from the given file.
    let load path =
        let value =
            use stream = new FileStream(path, FileMode.Open)
            JsonSerializer.Deserialize<Json>(stream)
        let encodingName = value.EncodingName
        let encoding = Encoding.Get(encodingName)
        let bigKeys = set value.Keys
        assert(bigKeys.Count = value.Keys.Length)
        create value.EncodingName encoding bigKeys
