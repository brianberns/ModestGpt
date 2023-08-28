﻿namespace ModestGpt

open TorchSharp

module ModestGpt =

    /// Sets TorchSharp random seed.
    let setSeed seed =
        torch.manual_seed(seed) |> ignore
        torch.cuda.manual_seed_all(seed)

[<AutoOpen>]
module TorchExt =

    /// Creates an explicit scalar.
    /// https://github.com/dotnet/TorchSharp/issues/1073
    let scalar (x : float) = x.ToScalar()

    /// Replacement for PyTorch's "@" operator.
    let (@@) a b = torch.matmul(a, b)

    type torch.Tensor with

        /// Moves the tensor to the given device. Friendly F# syntax.
        member tensor.To(device : string) = tensor.``to``(device)

        /// Moves the tensor to the given device. Friendly F# syntax.
        member tensor.To(device : torch.Device) = tensor.``to``(device)

    open System.Runtime.CompilerServices

    /// Moves the module's parameters and buffers to the given device.
    ///  Friendly F# syntax.
    [<Extension>]
    type IModuleExt =
        [<Extension>]
        static member To<'mdule when 'mdule :> torch.nn.Module>(
            mdule : 'mdule, device : string) =
            mdule.``to``(device)

module Tuple2 =

    /// Swap's the tuple's items.
    let swap (a, b) =
        b, a

    /// Maps a function over the tuple.
    let map f (a, b) =
        f a, f b

    /// Creates a tuple from the given array.
    let ofArray = function
        | [| a; b |] -> a, b
        | array -> failwith $"Unexpected array length: {array.Length}"

module Tuple3 =

    /// Maps a function over the tuple.
    let map f (a, b, c) =
        f a, f b, f c

    /// Creates a tuple from the given array.
    let ofArray = function
        | [| a; b; c |] -> a, b, c
        | array -> failwith $"Unexpected array length: {array.Length}"
