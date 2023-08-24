namespace ModestGpt.Model

type ModelConfig =
    {
        NumEmbed : int
        NumHead : int
        BlockSize : int
        Dropout : float
        VocabSize : int
        NumLayer : int
    }
