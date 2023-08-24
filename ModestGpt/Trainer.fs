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
                    let pairs = mdule.named_parameters(recurse = false)
                    for struct (parmName, parm) in pairs do
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
        assert(
            (decayMap.Values |> Seq.concat |> Seq.length) =
                (model.named_parameters() |> Seq.length))

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
        printfn "%A" optim
        0
