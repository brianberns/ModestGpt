namespace ModestGpt.Model

open TorchSharp
open type torch

open ModestGpt

type FeedForward(config) as self =
    inherit BaseModule("FeedForward")

    let size = 4 * config.NumEmbed
    let sequential =
        nn.Sequential(
            new Linear(config.NumEmbed, size),                       // to-do: clarify "c_fc" name
            nn.GELU(),                                               // activation layer
            new Projection(size, config))                              // to-do: clarify "residual" dropout

    do self.RegisterComponents()

    override _.forward(inp) = inp --> sequential

/// Transformer decoder block.
type TransformerBlock(config) as self =
    inherit BaseModule("TransformerBlock")

    let layer1 =
        nn.Sequential(
            new LayerNorm(config.NumEmbed),
            new CausalSelfAttention(config))
    let layer2 =
        nn.Sequential(
            new LayerNorm(config.NumEmbed),
            new FeedForward(config))

    do self.RegisterComponents()

    override _.forward(inp) =
        let x = inp + (inp --> layer1)
        x + (x --> layer2)

type Transformer(config) as self =
    inherit BaseModule("Transformer")

    let wte = new Embedding(config.VocabSize, config.NumEmbed)
    let wpe = new Embedding(config.BlockSize, config.NumEmbed)
    let sequential =
        let blocks =
            Array.init config.NumLayer (fun _ ->
                new TransformerBlock(config) :> BaseModule)
        nn.Sequential(
            nn.Dropout(config.Dropout),
            nn.Sequential(blocks),
            new LayerNorm(config.NumEmbed))

    do self.RegisterComponents()

    override _.forward(inp) =
        let device = inp.device
        let b, t = Tuple2.ofArray inp.shape
        if t > config.BlockSize then
            failwith $"Cannot forward sequence of length {t}, block size is only {config.BlockSize}"
        let pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) // shape (1, t)

        let tok_emb = inp --> wte // token embeddings of shape (b, t, n_embd)
        let pos_emb = pos --> wpe // position embeddings of shape (1, t, n_embd)
        (tok_emb + pos_emb) --> sequential
