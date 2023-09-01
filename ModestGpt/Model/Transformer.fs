namespace ModestGpt

open TorchSharp
open type torch

open ModestGpt

/// Multi-layer perceptron.
type FeedForward(config) as self =
    inherit BaseModule("FeedForward")

    let sequential =
        let size = 4 * config.NumEmbed                       // increase dimensionality by 4x
        nn.Sequential(
            new Linear(config.NumEmbed, size),               // fully-connected layer ("fc")
            nn.GELU(),                                       // activation layer
            new Projection(size, config.NumEmbed, config))   // project back to original dimensionality

    do self.RegisterComponents()

    override _.forward(inp) = inp --> sequential

/// Transformer decoder block.
type TransformerBlock(config) as self =
    inherit BaseModule("TransformerBlock")

    let attn =
        nn.Sequential(
            new LayerNorm(config.NumEmbed),
            new CausalSelfAttention(config))
    let mlp =
        nn.Sequential(
            new LayerNorm(config.NumEmbed),
            new FeedForward(config))

    do self.RegisterComponents()

    override _.forward(inp) =
        let x = inp + (inp --> attn)   // residual ("skip") connection at both layers
        x + (x --> mlp)

/// Entire transformer.
type Transformer(config) as self =
    inherit BaseModule("Transformer")

        // token embeddings
    let wte = new Embedding(config.VocabSize, config.NumEmbed)

        // position embeddings
    let wpe = new Embedding(config.BlockSize, config.NumEmbed)

    let sequential =
        let blocks =
            Array.init config.NumLayer (fun _ ->
                new TransformerBlock(config) :> BaseModule)
                |> nn.Sequential
        nn.Sequential(
            nn.Dropout(config.Dropout),
            blocks,
            new LayerNorm(config.NumEmbed))

    do self.RegisterComponents()

    override _.forward(inp) =

            // get sequence length
        let _b, t = Tuple2.ofArray inp.shape
        if t > config.BlockSize then
            failwith $"Cannot forward sequence of length {t}, block size is only {config.BlockSize}"

            // position indexes of shape (1, t)
        let pos =
            torch.arange(
                0, t,
                dtype = torch.long,
                device = inp.device)
                .unsqueeze(0)

        let tokEmb = inp --> wte   // token embeddings of shape (b, t, numEmbed)
        let posEmb = pos --> wpe   // position embeddings of shape (1, t, numEmbed)
        (tokEmb + posEmb) --> sequential
