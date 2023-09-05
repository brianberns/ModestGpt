namespace ModestGpt

open System

open TorchSharp
open type torch
open type TensorIndex
open FSharp.Core.Operators   // reclaim "float" and other F# operators

open ModestGpt

/// GPT Language Model
type Gpt(config) as self =
    inherit BaseModule("Gpt")

    let transformer = new Transformer(config)

        // Language model head. In the real GPT-2, this module's parameters are
        // the mirror image of (and thus shared with) the transformer's token
        // embeddings (wte). I'm not sure why minGPT doesn't do the same.
        // See https://huggingface.co/transformers/v2.0.0/_modules/transformers/modeling_gpt2.html.
    let lmHead = new Linear(config.NumEmbed, config.VocabSize, hasBias = false)

    do self.RegisterComponents()

    /// Produces logits for the given input.
    override _.forward(input) =
        input --> transformer --> lmHead

    /// Calculates loss for the given input compared to the given targets.
    member _.GetLoss(input, target : Tensor) =
        let logits = self.forward(input)
        nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target.view(-1),
            ignore_index = -1)

    /// Take a conditioning sequence (long tensor of shape (b,t)) and complete
    /// the sequence maxNewTokens times, feeding the predictions back into the
    /// model each time. Most likely you'll want to make sure to be in eval()
    /// mode of operation for this.
    member _.Generate(inp : Tensor, maxNewTokens, ?temperature, ?sample, ?topK) =

        let temperature = defaultArg temperature 1.0
        let sample = defaultArg sample false

        using (torch.no_grad()) (fun _ ->
            (inp.alias(), [1 .. maxNewTokens])
                ||> Seq.fold (fun inp _ ->

                    use _scope = torch.NewDisposeScope()
                    use inp = inp

                        // if the sequence context is growing too long we must crop it at block size
                    let inpCond =
                        if inp.size(1) <= config.BlockSize then inp
                        else inp[Colon, Slice(-config.BlockSize)]

                        // forward the model to get the logits for the index in the sequence
                    let logits = self.forward(inpCond)

                        // pluck the logits at the final step and scale by desired temperature
                    let logits = logits[Colon, Single(-1), Colon] / (scalar temperature)

                        // optionally crop the logits to only the top k options
                    Option.iter (fun topK ->
                        let v = torch.topk(logits, topK) |> fstv
                        logits[torch.lt(logits, v[Colon, Single(-1)])] <- Double.NegativeInfinity)
                        topK

                        // apply softmax to convert logits to (normalized) probabilities
                    let probs = softmax(logits, dim = -1)

                        // either sample from the distribution or take the most likely element
                    let inpNext =
                        if sample then
                            torch.multinomial(probs, num_samples=1)
                        else
                            torch.topk(probs, k = 1, dim = -1) |> sndv

                        // append sampled index to the running sequence and continue
                    torch.cat([| inp; inpNext |], dim = 1)
                        .DetachFromDisposeScope()))
