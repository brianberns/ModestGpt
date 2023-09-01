namespace ModestGpt

(*
 * This file contains wrappers around typical TorchSharp modules.
 * These wrappers provide support for GPT initialization and weight
 * decay.
 *)

open TorchSharp
open type torch
open TorchSharp.Modules
open FSharp.Core.Operators   // reclaim "float" and other F# operators

open ModestGpt

/// Module that takes a tensor as input and produces another tensor
/// as output.
type BaseModule = nn.Module<Tensor, Tensor>

module Init =

    /// Initializes the given using a normal distribution.
    let normal std tensor =
        nn.init.normal_(
            tensor,
            mean = 0.0,
            std = std) |> ignore

    /// Initializes the given tensor with zeros.
    let zeros tensor =
        nn.init.zeros_(tensor) |> ignore

    /// Initializes the given tensor with ones.
    let ones tensor =
        nn.init.ones_(tensor) |> ignore

/// Interface for specifying which of a module's parameters have weight
/// decay.
type IWeightDecay =
    abstract member ParameterSettings : seq<Parameter * bool> with get

module WeightDecay =

    /// Answers standard weight decay settings for a linear module.
    let ofLinear (linear : Linear) =
        seq {
            linear.weight, true
            if not (isNull linear.bias) then
                linear.bias, false
        }

/// Linear transformation.
type Linear(inputSize, outputSize, ?hasBias) as self =
    inherit BaseModule("Linear")

    let hasBias = defaultArg hasBias true
    let linear = nn.Linear(inputSize, outputSize, hasBias)

    do
        self.RegisterComponents()
        Init.normal 0.02 linear.weight
        if hasBias then Init.zeros linear.bias

    interface IWeightDecay with
        member _.ParameterSettings with get() =
            WeightDecay.ofLinear linear

    override _.forward(inp) = inp --> linear

/// Lookup table of dimensional embeddings.
type Embedding(size, numEmbed) as self =
    inherit BaseModule("Linear")

    let embedding = nn.Embedding(size, numEmbed)

    do
        self.RegisterComponents()
        Init.normal 0.02 embedding.weight

    interface IWeightDecay with
        member _.ParameterSettings
            with get() =
                seq { embedding.weight, false }

    override _.forward(inp) = inp --> embedding

/// Layer normalization.
type LayerNorm(shape : int64) as self =
    inherit BaseModule("LayerNorm")

    let layerNorm = nn.LayerNorm(shape)

    do
        self.RegisterComponents()
        Init.ones layerNorm.weight
        Init.zeros layerNorm.bias

    interface IWeightDecay with
        member _.ParameterSettings
            with get() =
                seq {
                    layerNorm.weight, false
                    layerNorm.bias, false
                }

    override _.forward(inp) = inp --> layerNorm
