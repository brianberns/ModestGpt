namespace ModestGpt.Model

open TorchSharp
open type torch
open FSharp.Core.Operators   // reclaim "float" and other F# operators

open ModestGpt

/// GPT Language Model
type Gpt(config) as self =
    inherit nn.Module<Tensor, Tensor, Tensor * Tensor>("GPT")

    let transformer = new Transformer(config)
    let lm_head = new Linear(config.NumEmbed, config.VocabSize, hasBias = false)

    do self.RegisterComponents()

    member _.forward(inp) =
        inp --> transformer --> lm_head

    override _.forward(inp, targets) =

        // forward the GPT model itself
        let logits = self.forward(inp)

        // calculate the loss
        let loss =
            nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index = -1)

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
