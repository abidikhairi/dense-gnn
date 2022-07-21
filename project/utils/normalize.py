import torch as th


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = th.sum(mx, dim=1)
    rowsum_invrt = th.pow(rowsum, -1)
    rowsum_invrt[th.isinf(rowsum_invrt)] = 0.

    mx_norm = th.matmul(th.diag(rowsum_invrt), mx)
    
    return mx_norm
