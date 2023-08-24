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
        let decayMap =
            seq {
                for mdule in model.modules() do
                    for struct (parmName, parm) in mdule.named_parameters() do
                        match parmName with
                            | EndsWith("bias") -> parm, false
                            | EndsWith("weight") ->
                                match mdule with
                                    | :? Modules.Linear -> parm, true
                                    | :? Modules.LayerNorm
                                    | :? Modules.Embedding -> parm, false
                                    | _ -> failwith "Unexpected"
                            | _ -> failwith "Unexpected"
            }
                |> Seq.groupBy snd
                |> Seq.map (fun (decay, group) ->
                    let parms =
                        group
                            |> Seq.map fst
                            |> Seq.toArray
                    decay, parms)
                |> Map

        // create the pytorch optimizer object
        let parmGroups =
            seq {
                AdamW.ParamGroup(
                    decayMap[true],
                    AdamW.Options(weight_decay = config.WeightDecay))
                AdamW.ParamGroup(
                    decayMap[false],
                    AdamW.Options(weight_decay = 0.0))
            }
        torch.optim.AdamW(
            parmGroups,
            config.LearningRate,
            config.Beta1,
            config.Beta2)
