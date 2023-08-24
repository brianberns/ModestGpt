namespace ModestGpt

open TorchSharp
open TorchSharp.Modules
open type torch
open type TensorIndex
open FSharp.Core.Operators   // reclaim "float" and other F# operators

type TrainerConfig =
    {
        Device : string
        NumWorkers : int
        MaxIters : int
        BatchSize : int
        LearningRate : float
        Beta1 : float
        Beta2 : float
        WeightDecay : float
        GradNormClip : float
    }

module Trainer =

    let private (|EndsWith|_|) (needle : string) (haystack : string) =
        if haystack.EndsWith(needle) then Some ()
        else Option.None

    let createOptimizer (model : Gpt) config =

        // separate out all parameters to those that will and won't experience regularizing weight decay
        let parmGroups =
            [|
                for mdule in model.modules() do
                    match mdule :> obj with
                        | :? IWeightDecay as module' ->
                            for parm, setting in module'.ParameterSettings do
                                printfn $"{mdule} {parm}: {setting}"
                                AdamW.ParamGroup(
                                    [ parm ],
                                    AdamW.Options(
                                        weight_decay =
                                            if setting then config.WeightDecay
                                            else 0.0))
                        | _ -> ()
            |]
        assert(parmGroups.Length = 
            (model.named_parameters() |> Seq.length))
        torch.optim.AdamW(
            parmGroups,
            config.LearningRate,
            config.Beta1,
            config.Beta2)

    [<EntryPoint>]
    let main args =
        use model =
            new Gpt {
                NumLayer = 3
                NumHead = 3
                NumEmbed =  48
                VocabSize = 3
                BlockSize = 6 * 2 - 1
                Dropout = 0.1
            }
        let optim =
            createOptimizer model
                {
                    Device = "auto"
                    NumWorkers = 4
                    MaxIters = 0
                    BatchSize = 64
                    LearningRate = 3e-4
                    Beta1 = 0.9
                    Beta2 = 0.95
                    WeightDecay = 0.1 // only applied on matmul weights
                    GradNormClip = 1.0
                }
        0
