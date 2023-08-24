namespace ModestGpt

open TorchSharp
open type torch
open TorchSharp.Modules
open FSharp.Core.Operators   // reclaim "float" and other F# operators

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

        nn.init.normal_(
            linear.weight,
            mean = 0.0,
            std = 0.02) |> ignore

        if hasBias then
            nn.init.zeros_(linear.bias) |> ignore
        else assert(isNull linear.bias)

    interface IWeightDecay with
        member _.ParameterSettings with get() =
            WeightDecay.ofLinear linear

    override _.forward(inp) = inp --> linear

type Embedding(size, numEmbed) as self =
    inherit BaseModule("Linear")

    let embedding = nn.Embedding(size, numEmbed)

    do
        self.RegisterComponents()

        nn.init.normal_(
            embedding.weight,
            mean = 0.0,
            std = 0.02) |> ignore

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

        nn.init.ones_(layerNorm.weight) |> ignore
        nn.init.zeros_(layerNorm.bias) |> ignore

    interface IWeightDecay with
        member _.ParameterSettings
            with get() =
                seq {
                    layerNorm.weight, false
                    layerNorm.bias, false
                }

    override _.forward(inp) = inp --> layerNorm
