namespace ModestGpt

open System

open TorchSharp
open type torch
open type TensorIndex
open FSharp.Core.Operators   // reclaim "float" and other F# operators

open ModestGpt

/// GPT Language Model
type Gpt(config) as self =
    inherit nn.Module<Tensor, Tensor, Tensor * Tensor>("Gpt")

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

    /// Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    /// the sequence maxNewTokens times, feeding the predictions back into the model each time.
    /// Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    member _.Generate(idx : Tensor, maxNewTokens, ?temperature, ?sample, ?topK) =
        let temperature = defaultArg temperature 1.0
        let sample = defaultArg sample false
        using (torch.no_grad()) (fun _ ->
            (idx.alias(), [1.. maxNewTokens])
                ||> Seq.fold (fun idx _ ->
                    use _scope = torch.NewDisposeScope()
                    use idx = idx
                    // if the sequence context is growing too long we must crop it at BlockSize
                    let idxCond =
                        if idx.size(1) <= config.BlockSize then idx
                        else idx[Colon, Slice(-config.BlockSize)]
                    // forward the model to get the logits for the index in the sequence
                    let logits = self.forward(idxCond)
                    // pluck the logits at the final step and scale by desired temperature
                    let logits = logits[Colon, Single(-1), Colon] / (scalar temperature)
                    // optionally crop the logits to only the top k options
                    Option.iter (fun topK ->
                        let struct (v, _) = torch.topk(logits, topK)
                        logits[torch.lt(logits, v[Colon, Single(-1)])] <- Double.NegativeInfinity)
                        topK
                    // apply softmax to convert logits to (normalized) probabilities
                    let probs = softmax(logits, dim = -1)
                    // either sample from the distribution or take the most likely element
                    let idxNext =
                        if sample then
                            torch.multinomial(probs, num_samples=1)
                        else
                            let struct (_, idxNext) = torch.topk(probs, k=1, dim = -1)
                            idxNext
                    // append sampled index to the running sequence and continue
                    torch.cat([|idx; idxNext|], dim=1).DetachFromDisposeScope()))
