import torch
import genesis as gs

def create_linear_array(N):
    base_seq = torch.arange(1, N + 1, dtype=gs.tc_float)
    out = base_seq / base_seq.sum()
    return out

def create_exp_array(N, base=1.1):
    exponents = torch.arange(N, dtype=gs.tc_float)
    base_seq = base ** exponents
    out = base_seq / base_seq.sum()
    return out

def create_custom_array(N):
    base_seq = torch.ones(N, dtype=gs.tc_float)
    base_seq[:N-1] = 0.5 / (N - 1)
    base_seq[N-1] = 0.5
    # assert base_seq.sum() == 1.0
    base_seq = base_seq / base_seq.sum() # ensure sum to 1
    return base_seq
