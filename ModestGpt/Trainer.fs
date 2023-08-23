namespace ModestGpt

type TrainerConfig =
    {
        Device : string
        NumWorkers : int
        MaxIters : int
        BatchSize : int
        LearningRate : float
        Betas : float * float
        WeightDecay : float
        GradNormClip : float
    }

module Trainer =

    let createOptimizer model config =
