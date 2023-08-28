namespace ModestGpt

open System

open TorchSharp
open type torch
open type utils.data
open FSharp.Core.Operators   // reclaim "float" and other F# operators

type Dataset = Dataset<Tensor * Tensor>

/// Minimal data loader.
type DataLoader(dataset : Dataset, batchSize, ?shuffle, ?numWorker, ?dropLast) =
    inherit DataLoader<Tensor * Tensor, Tensor * Tensor>(
        dataset,
        batchSize,
        DataLoader.Collate,
        ?shuffle = shuffle,
        ?num_worker = numWorker,
        ?drop_last = dropLast)

    /// Extracts and merges tensors from the given items.
    static let collate f items (device : Device) =
        let tensors =
            items
                |> Seq.map (fun item ->
                    let (tensor : torch.Tensor) = f item
                    tensor.unsqueeze(0))
                |> Seq.toArray
        let tensor = torch.cat(tensors, 0)
        if tensor.device_type <> device.``type``
            || tensor.device_index <> device.index then
            tensor.To(device)
        else tensor

    /// Extracts and merges tensors from the given pairs.
    static member private Collate =
        Func<_, _, _>(fun pairs device ->
            let pairs = Seq.cache pairs
            collate fst pairs device,
            collate snd pairs device)

    /// Adds fractional epoch to the loader's iterator.
    member this.Indexed =
        Seq.indexed this
            |> Seq.map (fun (iBatch, (x, y)) ->
                let epochFrac =
                    float (iBatch * batchSize) / float dataset.Count
                epochFrac, x, y)
