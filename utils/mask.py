import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_attn_subsequence_mask(seq):
    """
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 不包括对角线的上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).bool()
    return subsequence_mask.to(device)  # [batch_size, tgt_len, tgt_len]
