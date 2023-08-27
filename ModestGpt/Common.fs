namespace ModestGpt

open TorchSharp

module ModestGpt =

    /// Sets random seed.
    let setSeed seed =
        torch.manual_seed(seed) |> ignore
        torch.cuda.manual_seed_all(seed)

[<AutoOpen>]
module TorchExt =

    open System.Runtime.CompilerServices

    let scalar (x : float) = x.ToScalar()

    let (@@) a b = torch.matmul(a, b)

    type torch.Tensor with
        member tensor.To(device : string) = tensor.``to``(device)
        member tensor.To(device : torch.Device) = tensor.``to``(device)

    [<Extension>]
    type IModuleExt =
        [<Extension>]
        static member To<'mdule when 'mdule :> torch.nn.Module>(
            mdule : 'mdule, device : string) =
            mdule.``to``(device)

module Tuple2 =

    let swap (a, b) =
        b, a

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
