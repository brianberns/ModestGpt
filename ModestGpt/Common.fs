namespace ModestGpt

open TorchSharp
open type torch

module ModestGpt =

    /// Sets random seed.
    let setSeed seed =
        torch.manual_seed(seed) |> ignore
        torch.cuda.manual_seed_all(seed)

[<AutoOpen>]
module TorchExt =

    let scalar (x : float) = x.ToScalar()

    let (@@) a b = torch.matmul(a, b)

module Tuple2 =

    let map f (a, b) =
        f a, f b

    let ofArray = function
        | [| a; b |] -> a, b
        | array -> failwith $"Unexpected array length: {array.Length}"

module Tuple3 =

    let map f (a, b, c) =
        f a, f b, f c

    let ofArray = function
        | [| a; b; c |] -> a, b, c
        | array -> failwith $"Unexpected array length: {array.Length}"
