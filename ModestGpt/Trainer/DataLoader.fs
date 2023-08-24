namespace ModestGpt.Trainer

open System

open TorchSharp
open type torch
open type utils.data

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

    static let collate f items (device : Device) =
        let tensors =
            items
                |> Seq.map (fun item ->
                    let (tensor : torch.Tensor) = f item
                    tensor.unsqueeze(0))
                |> Seq.toArray
        let tensor = torch.cat(tensors, 0)
        if tensor.device_type <> device.``type`` || tensor.device_index <> device.index then
            tensor.``to``(device)
        else tensor

    static member private Collate =
        Func<_, _, _>(fun pairs device ->
            let pairs = Seq.cache pairs
            collate fst pairs device,
            collate snd pairs device)
