namespace ModestGpt.Trainer

open System

open TorchSharp
open type torch
open type utils.data

type MinDataset = utils.data.Dataset<Tensor * Tensor>

/// Minimal data loader.
type DataLoader(dataset : MinDataset, batch_size, ?shuffle, ?num_worker, ?drop_last) =
    inherit DataLoader<Tensor * Tensor, Tensor * Tensor>(
        dataset,
        batch_size,
        DataLoader.Collate,
        ?shuffle = shuffle,
        ?num_worker = num_worker,
        ?drop_last = drop_last)

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
