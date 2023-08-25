namespace ModestGpt.Model

type Config =
    {
        NumEmbed : int
        NumHead : int
        BlockSize : int
        Dropout : float
        VocabSize : int
        NumLayer : int
    }
