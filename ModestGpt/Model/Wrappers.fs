namespace ModestGpt

open TorchSharp
open type torch
open TorchSharp.Modules
open FSharp.Core.Operators   // reclaim "float" and other F# operators

open ModestGpt

type BaseModule = nn.Module<Tensor, Tensor>

module Init =

    let normal std tensor =
        nn.init.normal_(
            tensor,
            mean = 0.0,
            std = std) |> ignore

    let zeros tensor =
        nn.init.zeros_(tensor) |> ignore

    let ones tensor =
        nn.init.ones_(tensor) |> ignore

type IWeightDecay =
    abstract member ParameterSettings : seq<Parameter * bool> with get

module WeightDecay =

    let ofLinear (linear : Linear) =
        seq {
            linear.weight, true
            if not (isNull linear.bias) then
                linear.bias, false
        }

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
