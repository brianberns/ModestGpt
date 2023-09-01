namespace ModestGpt

open TorchSharp
open type torch
open FSharp.Core.Operators   // reclaim "float" and other F# operators

open ModestGpt

type Projection(inputSize, config) as self =
    inherit BaseModule("Projection")

    let linear = nn.Linear(inputSize, config.NumEmbed)
    let dropout = nn.Dropout(config.Dropout)

    do
        self.RegisterComponents()

           // apply a special scaled init to the residual projections, per GPT-2 paper
        let std = 0.02 / sqrt (2.0 * float config.NumLayer)
        Init.normal std linear.weight
        Init.zeros linear.bias

    interface IWeightDecay with
        member _.ParameterSettings with get() =
            WeightDecay.ofLinear linear

    override _.forward(inp) = inp --> linear --> dropout
