namespace ModestGpt

open TorchSharp
open type torch
open FSharp.Core.Operators   // reclaim "float" and other F# operators

type Projection(inputSize, config) as self =
    inherit BaseModule("Projection")

    let linear = nn.Linear(inputSize, config.NumEmbed)
    let dropout = nn.Dropout(config.Dropout)

    do
        self.RegisterComponents()

        let std = 0.02 / sqrt (2.0 * float config.NumLayer)   // apply a special scaled init to the residual projections, per GPT-2 paper

        nn.init.normal_(
            linear.weight,
            mean = 0.0,
            std = std) |> ignore

        nn.init.zeros_(linear.bias) |> ignore

    interface IWeightDecay with
        member _.ParameterSettings with get() =
            WeightDecay.ofLinear linear

    override _.forward(inp) = inp --> linear --> dropout
