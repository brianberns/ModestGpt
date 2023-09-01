namespace ModestGpt

/// Model configuration.
type ModelConfig =
    {
        /// Number of tokens known the the model.
        VocabSize : int

        /// Maximum number of tokens in a sample.
        BlockSize : int

        /// Dimensionality of token embeddings.
        NumEmbed : int

        /// Number of stacked transformer blocks.
        NumLayer : int

        /// Number of attention heads.
        NumHead : int

        /// Dropout probability.
        Dropout : float
    }
