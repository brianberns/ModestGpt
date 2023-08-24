namespace ModestGpt

open System

open TorchSharp
open TorchSharp.Modules
open type torch
open type TensorIndex
open FSharp.Core.Operators   // reclaim "float" and other F# operators

[<AutoOpen>]
module TorchExt =

    let scalar (x : float) = x.ToScalar()

    let (@) a b = torch.matmul(a, b)

module Tuple2 =

    let map f (a, b) =
        f a, f b

    let ofArray = function
        | [| a; b |] -> a, b
        | array -> failwith $"Unexpected array length: {array.Length}"

module Tuple3 =

    let map f (a, b, c) =
        f a, f b, f c

    let ofArray = function
        | [| a; b; c |] -> a, b, c
        | array -> failwith $"Unexpected array length: {array.Length}"

type ModelConfig =
    {
        NumEmbed : int
        NumHead : int
        BlockSize : int
        Dropout : float
        VocabSize : int
        NumLayer : int
    }

type BaseModule = nn.Module<Tensor, Tensor>

type Projection(inputSize, outputSize, dropout) as self =
    inherit BaseModule("Projection")

    let linear = nn.Linear(inputSize, outputSize)
    let sequential =
        nn.Sequential(
            linear,
            nn.Dropout(dropout))

    do self.RegisterComponents()

    member internal _.Linear = linear

    override _.forward(inp) = inp --> sequential

/// Causal: only looks at previous tokens.
type CausalSelfAttention(config) as self =
    inherit BaseModule("CausalSelfAttention")

    let blockSize = config.BlockSize
    do assert(config.NumEmbed % config.NumHead = 0)

        // query, key, value projections for all heads, but in a batch
    let inpProj = nn.Linear(config.NumEmbed, 3L * int64 config.NumEmbed)

        // causal mask to ensure that attention is only applied to the left in the input sequence
    let bias =
        (torch.ones(blockSize, blockSize)
            |> torch.tril)
            .view(1, 1, blockSize, blockSize)

        // regularization
    let dropout = nn.Dropout(config.Dropout)

        // output projection
    let outProj = new Projection(config.NumEmbed, config.NumEmbed, config.Dropout)

    do self.RegisterComponents()

    override _.forward(inp) =

            // batch size, sequence length, embedding dimensionality (numEmbed)
        let B, T, C = Tuple3.ofArray inp.shape
        assert(C = config.NumEmbed)

            // calculate query, key, values for all heads in batch and move head forward to be the batch dim
        let query, key, value =
            (inp --> inpProj)
                .split(config.NumEmbed, dim = 2)
                |> Tuple3.ofArray
                |> Tuple3.map (fun t ->
                    t.view(B, T, config.NumHead, C / int64 config.NumHead)
                        .transpose(1, 2)) // (B, nh, T, hs)

            // causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        let att =
            let att =
                let scale = 1.0 / (key.size(-1) |> float |> sqrt)
                (query @ key.transpose(-2, -1)) * scalar scale
            let mask =
                let slice = Slice(stop = T)
                bias[Colon, Colon, slice, slice]
            att.masked_fill(torch.eq(mask, 0), Double.NegativeInfinity)
                .softmax(dim = -1)
                --> dropout

             // (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        (att @ value)
            .transpose(1, 2)
            .contiguous()
            .view(B, T, C) // re-assemble all head outputs side by side
            --> outProj   // output projection

type FeedForward(config) as self =
    inherit BaseModule("FeedForward")

    let size = 4 * config.NumEmbed
    let sequential =
        nn.Sequential(
            nn.Linear(config.NumEmbed, size),                       // to-do: clarify "c_fc" name
            nn.GELU(),                                               // activation layer
            new Projection(size, config.NumEmbed, config.Dropout))                              // to-do: clarify "residual" dropout

    do self.RegisterComponents()

    override _.forward(inp) = inp --> sequential

/// Transformer decoder block.
type TransformerBlock(config) as self =
    inherit BaseModule("TransformerBlock")

    let layer1 =
        nn.Sequential(
            nn.LayerNorm(config.NumEmbed),
            new CausalSelfAttention(config))
    let layer2 =
        nn.Sequential(
            nn.LayerNorm(config.NumEmbed),
            new FeedForward(config))

    do self.RegisterComponents()

    override _.forward(inp) =
        let x = inp + (inp --> layer1)
        x + (x --> layer2)

type Transformer(config) as self =
    inherit BaseModule("Transformer")

    let wte = nn.Embedding(config.VocabSize, config.NumEmbed)
    let wpe = nn.Embedding(config.BlockSize, config.NumEmbed)
    let sequential =
        let blocks =
            Array.init config.NumLayer (fun _ ->
                new TransformerBlock(config) :> BaseModule)
        nn.Sequential(
            nn.Dropout(config.Dropout),
            nn.Sequential(blocks),
            nn.LayerNorm(config.NumEmbed))

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

/// GPT Language Model
type Gpt(config) as self =
    inherit nn.Module<Tensor, Tensor, Tensor * Tensor>("GPT")

    let transformer = new Transformer(config)
    let lm_head = nn.Linear(config.NumEmbed, config.VocabSize, hasBias = false)

    let initWeight : nn.Module -> _ = function

        | :? Projection as proj ->   // special scaled init to the residual projections, per GPT-2 paper
            let linear = proj.Linear
            nn.init.normal_(
                linear.weight,
                mean = 0.0,
                std = 0.02 / sqrt (2.0 * float config.NumLayer)) |> ignore
            linear.bias |> nn.init.zeros_ |> ignore

        | :? Linear as linear ->
            nn.init.normal_(linear.weight, mean = 0.0, std = 0.02) |> ignore
            if isNull linear.bias |> not then
                linear.bias |> nn.init.zeros_ |> ignore

        | :? Embedding as embedding ->
            nn.init.normal_(embedding.weight, mean = 0.0, std = 0.02) |> ignore

        | :? LayerNorm as norm ->
            norm.bias |> nn.init.zeros_ |> ignore
            norm.weight |> nn.init.ones_ |> ignore

        | _ -> ()

    do
        self.RegisterComponents()
        self.apply(initWeight) |> ignore

    member _.forward(idx) =
        idx --> transformer --> lm_head

    override _.forward(idx, targets) =

        // forward the GPT model itself
        let logits = self.forward(idx)

        // calculate the loss
        let loss =
            nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index = -1)

        logits, loss

    (*
    /// Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    /// the sequence max_new_tokens times, feeding the predictions back into the model each time.
    /// Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    member _.generate(idx : Tensor, max_new_tokens, ?temperature, ?do_sample, ?top_k) =
        let temperature = defaultArg temperature 1.0
        let do_sample = defaultArg do_sample false
        using (torch.no_grad()) (fun _ ->
            (idx.alias(), range(max_new_tokens))
                ||> Seq.fold (fun idx _ ->
                    use _scope = torch.NewDisposeScope()
                    use idx = idx
                    // if the sequence context is growing too long we must crop it at block_size
                    let idx_cond =
                        if idx.size(1) <= config.block_size then idx
                        else idx[Colon, Slice(-config.block_size)]
                    // forward the model to get the logits for the index in the sequence
                    let logits = self.forward(idx_cond)
                    // pluck the logits at the final step and scale by desired temperature
                    let logits = logits[Colon, Single(-1), Colon] / (temperature.ToScalar())
                    // optionally crop the logits to only the top k options
                    Option.iter (fun top_k ->
                        let struct (v, _) = torch.topk(logits, top_k)
                        logits[torch.lt(logits, v[Colon, Single(-1)])] <- Double.NegativeInfinity)
                        top_k
                    // apply softmax to convert logits to (normalized) probabilities
                    let probs = softmax(logits, dim = -1)
                    // either sample from the distribution or take the most likely element
                    let idx_next =
                        if do_sample then
                            torch.multinomial(probs, num_samples=1)
                        else
                            let struct (_, idx_next) = torch.topk(probs, k=1, dim = -1)
                            idx_next
                    // append sampled index to the running sequence and continue
                    torch.cat([|idx; idx_next|], dim=1).DetachFromDisposeScope()))
    *)
