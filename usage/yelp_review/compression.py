import torch


def compress_embeddings(embeddings, lens):
    """Here the logic is the following: if a token (and therefore its embedding)
    is not attended, its value is irrelevant
    """
    compressed_data = []
    for tmp, data_len in zip(embeddings, lens):
        useful_data = tmp[:data_len]
        compressed_data.append(useful_data)
    return compressed_data


def uncompress_embeddings(compressed_embeddings):
    reconstructed, lens = [], []
    for tmp in compressed_embeddings:
        lens.append(tmp.shape[0])
        a = torch.vstack([tmp, torch.zeros(512 - len(tmp), 768)])[None,:,:]
        reconstructed.append(a)
    reconstructed = torch.vstack(reconstructed)
    return reconstructed, lens


def get_extended_attention_mask(batch_size, max_context_size, lens):
    if not isinstance(lens, torch.Tensor):
        lens = torch.tensor(lens, dtype=torch.int16)
    tmp = torch.arange(max_context_size).expand(batch_size, max_context_size)
    attention_mask = 1.0 - (tmp < lens.unsqueeze(1)).int()
    attention_mask = attention_mask * torch.finfo(torch.float32).min
    attention_mask = attention_mask[:, None, None, :]
    return attention_mask
