namespace ModestGpt

open System

open TorchSharp
open type torch
open type TensorIndex
open FSharp.Core.Operators   // reclaim "float" and other F# operators

open ModestGpt

/// Multi-head masked self-attention.
type CausalSelfAttention(config) as self =
    inherit BaseModule("CausalSelfAttention")

    do assert(config.NumEmbed % config.NumHead = 0)

        // query, key, value projections for all heads, but in a batch
    let inpProj = new Linear(config.NumEmbed, 3L * int64 config.NumEmbed)

        // causal mask to ensure that attention is only applied to the left in the input sequence
    let bias =
        let blockSize = config.BlockSize
        (torch.ones(blockSize, blockSize)
            |> torch.tril)
            .view(1, 1, blockSize, blockSize)

        // regularization
    let dropout = nn.Dropout(config.Dropout)

        // output projection
    let outProj = new Projection(config.NumEmbed, config.NumEmbed, config)   // to-do: this doesn't change the dimensionality, so I'm not sure what it's for

    do self.RegisterComponents()

    override _.forward(inp) =

            // batch size, sequence length, embedding dimensionality (numEmbed)
        let B, T, C = Tuple3.ofArray inp.shape
        assert(T <= config.BlockSize)
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
                (query @@ key.transpose(-2, -1)) * scalar scale
            let mask =
                let slice = Slice(stop = T)
                bias[Colon, Colon, slice, slice]
            att.masked_fill(torch.eq(mask, 0), Double.NegativeInfinity)
                .softmax(dim = -1)
                --> dropout

             // (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        (att @@ value)
            .transpose(1, 2)
            .contiguous()
            .view(B, T, C)   // re-assemble all head outputs side by side
            --> outProj      // output projection
