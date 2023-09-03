namespace ModestGpt

open TorchSharp

module ModestGpt =

    /// Sets TorchSharp random seed.
    let setSeed seed =
        torch.manual_seed(seed) |> ignore
        torch.cuda.manual_seed_all(seed)

[<AutoOpen>]
module TorchExt =

    /// First item of a value tuple.
    let fstv (struct (x, _)) = x

    /// Second item of a value tuple.
    let sndv (struct (_, y)) = y

    /// Creates an explicit scalar.
    /// https://github.com/dotnet/TorchSharp/issues/1073
    let scalar (x : float) = x.ToScalar()

    /// Replacement for PyTorch's "@" operator.
    let (@@) a b = torch.matmul(a, b)

    type torch.Tensor with

        /// Friendly F# syntax for moving the tensor to the given device.
        member tensor.To(device : string) = tensor.``to``(device)

        /// Friendly F# syntax for moving the tensor to the given device.
        member tensor.To(device : torch.Device) = tensor.``to``(device)

    open System.Runtime.CompilerServices

    /// Friendly F# syntax for moving the module's parameters and buffers
    /// to the given device.
    [<Extension>]
    type IModuleExt =
        [<Extension>]
        static member To<'mdule when 'mdule :> torch.nn.Module>(
            mdule : 'mdule, device : string) =
            mdule.``to``(device)

module Seq =

    /// Answers the maximum value in the given sequence using the given
    /// projection, if any.
    let tryMaxBy projection (items : seq<_>) =
        use e = items.GetEnumerator()
        if e.MoveNext() then
            let mutable maxItem = e.Current
            let mutable maxValue = projection maxItem
            while e.MoveNext() do
                let value = projection e.Current
                if value > maxValue then
                    maxItem <- e.Current
                    maxValue <- value
            Some maxItem
        else None

module Tuple2 =

    /// Swaps the tuple's items.
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
