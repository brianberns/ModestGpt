﻿namespace ModestGpt

namespace MinGptSharp

open System

open TorchSharp
open type torch
open type TensorIndex
open FSharp.Core.Operators   // reclaim "float" and other F# operators

[<AutoOpen>]
module TorchExt =

    let scalar (x : float) = x.ToScalar()

    let (@) a b = torch.matmul(a, b)

module Tuple3 =

    let map f (a, b, c) =
        f a, f b, f c

    let ofArray = function
        | [| a; b; c |] -> a, b, c
        | array -> failwith $"Unexpected array length: {array.Length}"

module Model =

    let feedForward numEmbed dropoutProb =
        let size = 4 * numEmbed
        nn.Sequential(
            nn.Linear(numEmbed, size),   // to-do: clarify "c_fc" name
            nn.GELU(),                   // activation layer
            nn.Linear(size, numEmbed),   // to-do: clarify "c_proj" name
            nn.Dropout(dropoutProb))     // to-do: clarify "residual" dropout

/// Causal: only looks at previous tokens.
type CausalSelfAttention(numEmbed : int, numHead : int, blockSize : int, dropoutProb) as self =
    inherit nn.Module<Tensor, Tensor>("CausalSelfAttention")

    do assert(numEmbed % numHead = 0)

        // key, query, value projections for all heads, but in a batch
    let inpProj = nn.Linear(numEmbed, 3L * int64 numEmbed)

        // output projection
    let outProj =
        nn.Sequential(
            nn.Linear(numEmbed, numEmbed),
            nn.Dropout(dropoutProb))

        // regularization
    let dropout = nn.Dropout(dropoutProb)

        // causal mask to ensure that attention is only applied to the left in the input sequence
    let bias =
        torch
            .tril(torch.ones(blockSize, blockSize))
            .view(1, 1, blockSize, blockSize)

    do self.RegisterComponents()

    override _.forward(inp) =

        let B, T, C = inp.size() |> Tuple3.ofArray // batch size, sequence length, embedding dimensionality (numEmbed)

            // calculate query, key, values for all heads in batch and move head forward to be the batch dim
        let q, k, v =
            let prepare (t : Tensor) =
                t.view(B, T, numHead, C / int64 numHead).transpose(1, 2) // (B, nh, T, hs)
            (inp --> inpProj)
                .split(numEmbed, dim = 2)
                |> Tuple3.ofArray
                |> Tuple3.map prepare

            // causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        let att =
            let att =
                let scale = 1.0 / (k.size(-1) |> float |> sqrt)
                (q @ k.transpose(-2, -1)) * scalar scale
            let mask = bias[Colon, Colon, Slice(stop=T), Slice(stop=T)]
            att.masked_fill(torch.eq(mask, 0), Double.NegativeInfinity)
                .softmax(dim = -1)
                --> dropout
        (att @ v) // (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            .transpose(1, 2)
            .contiguous()
            .view(B, T, C) // re-assemble all head outputs side by side
            --> outProj   // output projection

/// Transformer decoder block.
type TransformerBlock(numEmbed : int, numHead : int, blockSize, dropoutProb) as self =
    inherit nn.Module<Tensor, Tensor>("TransformerBlock")

    do assert(numEmbed % numHead = 0)

    let layerNorm1 = nn.LayerNorm(numEmbed)
    let attention = new CausalSelfAttention(numEmbed, numHead, blockSize, dropoutProb)
    let layerNorm2 = nn.LayerNorm(numEmbed)
    let feedForward = Model.feedForward numEmbed dropoutProb

    do self.RegisterComponents()

    override _.forward(x) =
        let x = x + (x --> layerNorm1 --> attention)
        let x = x + (x --> layerNorm2 --> feedForward)
        x

/// This is a submodule of GPT in the original minGPT, but it works better as a
/// top-level module in F#.
type Transformer(config) as self =
    inherit nn.Module<Tensor, Tensor>("Transformer")

    let block_size = config.block_size

    let wte = nn.Embedding(config.vocab_size, config.n_embd)
    let wpe = nn.Embedding(config.block_size, config.n_embd)
    let drop = nn.Dropout(config.embd_pdrop)
    let h = nn.ModuleList([| for _ in range(config.n_layer) -> new Block(config) |])
    let ln_f = nn.LayerNorm(config.n_embd)

    do self.RegisterComponents()

    override _.forward(idx) =
        let device = idx.device
        let [| b; t; |] = idx.size()
        if t > block_size then
            failwith $"Cannot forward sequence of length {t}, block size is only {block_size}"
        let pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) // shape (1, t)

        let tok_emb = idx --> wte // token embeddings of shape (b, t, n_embd)
        let pos_emb = pos --> wpe // position embeddings of shape (1, t, n_embd)
        let x = (tok_emb + pos_emb) --> drop
        let x = Seq.fold (-->) x h
        x --> ln_f

/// GPT Language Model
type GPT(config) as self =
    inherit nn.Module<Tensor, Tensor, Tensor * Tensor>("GPT")

    do
        assert(config.vocab_size > 0)
        assert(config.block_size > 0)

    let type_given = String.IsNullOrWhiteSpace(config.model_type) |> not
    let params_given = config.n_layer > 0 && config.n_head > 0 && config.n_embd > 0
    do assert (type_given <> params_given) // exactly one of these (XOR)
    let config =
        if type_given then
            // translate from model_type to detailed configuration
            match config.model_type with
                // names follow the huggingface naming conventions
                // GPT-1
                | "openai-gpt" ->  { config with n_layer=12; n_head=12; n_embd= 768 } //  117M params
                // GPT-2 configs
                | "gpt2" ->        { config with n_layer=12; n_head=12; n_embd= 768 } //  124M params
                | "gpt2-medium" -> { config with n_layer=24; n_head=16; n_embd=1024 } //  350M params
                | "gpt2-large" ->  { config with n_layer=36; n_head=20; n_embd=1280 } //  774M params
                | "gpt2-xl"  ->    { config with n_layer=48; n_head=25; n_embd=1600 } // 1558M params
                // Gophers
                | "gopher-44m" ->  { config with n_layer= 8; n_head=16; n_embd= 512 }
                // (there are a number more...)
                // I made these tiny models up
                | "gpt-mini" ->    { config with n_layer= 6; n_head= 6; n_embd= 192 }
                | "gpt-micro" ->   { config with n_layer= 4; n_head= 4; n_embd= 128 }
                | "gpt-nano" ->    { config with n_layer= 3; n_head= 3; n_embd=  48 }
        else config

    let transformer = new Transformer(config)
    let lm_head = nn.Linear(config.n_embd, config.vocab_size, hasBias=false)

    let _init_weights(mdule : nn.Module) =
        match mdule with
            | :? Modules.Linear as linear ->
                torch.nn.init.normal_(linear.weight, mean=0.0, std=0.02) |> ignore
                if isNull linear.bias |> not then
                    torch.nn.init.zeros_(linear.bias) |> ignore
            | :? Modules.Embedding as embedding ->
                torch.nn.init.normal_(embedding.weight, mean=0.0, std=0.02) |> ignore
            | :? Modules.LayerNorm as norm ->
                torch.nn.init.zeros_(norm.bias) |> ignore
                torch.nn.init.ones_(norm.weight) |> ignore
            | _ -> ()

    do
        self.RegisterComponents()

        // init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(_init_weights) |> ignore
        for pn, p in self.named_parameters() do
            if pn.EndsWith("c_proj.weight") then
                torch.nn.init.normal_(p, mean=0.0, std=0.02/Math.Sqrt(2.0 * float config.n_layer)) |> ignore

        // report number of parameters (note we don't count the decoder parameters in lm_head)
        let n_params = Seq.sum [ for p in transformer.parameters() -> p.numel() ]
        printfn "number of parameters: %d" n_params

    static member get_default_config() =
        {
            // either model_type or (n_layer, n_head, n_embd) must be given in the config
            model_type = "gpt"
            n_layer = -1
            n_head = -1
            n_embd =  -1
            // these options must be filled in externally
            vocab_size = -1
            block_size = -1
            // dropout hyperparameters
            embd_pdrop = 0.1
            resid_pdrop = 0.1
            attn_pdrop = 0.1
        }

    /// This long function is unfortunately doing something very simple and is being very defensive:
    /// We are separating out all parameters of the model into two buckets: those that will experience
    /// weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    /// We are then returning the PyTorch optimizer object.
    member _.configure_optimizers(train_config) =

        // separate out all parameters to those that will and won't experience regularizing weight decay
        let mfpns =
            [|
                for mn, m in self.named_modules() do
                    for struct(pn, p) in m.named_parameters() do
                        m, $"{mn}.{pn}" // full param name
            |]
        let decay, no_decay =
            ((Set.empty, Set.empty), mfpns)
                ||> Seq.fold (fun (decay, no_decay) (m, fpn) ->
                // random note: because named_modules and named_parameters are recursive
                // we will see the same tensors p many many times. but doing it this way
                // allows us to know which parent module any tensor p belongs to...
                if fpn.EndsWith("bias") then
                    // all biases will not be decayed
                    decay, Set.add fpn no_decay
                elif fpn.EndsWith("weight") then
                    match m with
                        | :? Modules.Linear ->
                            // weights will be weight decayed
                            Set.add fpn decay, no_decay
                        | :? Modules.LayerNorm
                        | :? Modules.Embedding ->
                            // weights will NOT be weight decayed
                            decay, Set.add fpn no_decay
                        | _ -> decay, no_decay
                else decay, no_decay)

        // validate that we considered every parameter
        let param_dict = Map [ for struct (pn, p) in self.named_parameters() -> pn, p ]
        let inter_params = Set.intersect decay no_decay
        let union_params = Set.union decay no_decay
        assert (inter_params.Count = 0)
        assert (param_dict.Count = union_params.Count)

        // create the pytorch optimizer object
        let optim_groups =
            [
                Modules.AdamW.ParamGroup(
                    [ for pn in decay -> param_dict[pn] ],
                    Modules.AdamW.Options(weight_decay=train_config.weight_decay))
                Modules.AdamW.ParamGroup(
                    [ for pn in no_decay -> param_dict[pn] ],
                    Modules.AdamW.Options(weight_decay=0.0))
            ]
        let beta1, beta2 = train_config.betas
        let optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, beta1=beta1, beta2=beta2)
        optimizer

    member _.forward(idx) =
        idx --> transformer --> lm_head

    override _.forward(idx, targets) =

        // forward the GPT model itself
        let logits = self.forward(idx)

        // calculate the loss
        let loss =
            nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index = -1)

        logits, loss

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
