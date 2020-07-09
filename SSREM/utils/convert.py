import torch
from torch.autograd import Variable
PAD_ID, UNK_ID, SOS_ID, EOS_ID = [0, 1, 2, 3]


def to_var(x, on_cpu=False, gpu_id=None, async_var=False):
    if torch.cuda.is_available() and not on_cpu:
        x = x.cuda(gpu_id, async_var)
    return x


def to_tensor(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data


def pad(tensor, length):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
        else:
            return var
    else:
        if length > tensor.size(0):
            return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
        else:
            return tensor


def pad_and_pack(tensor_list):
    length_list = ([t.size(0) for t in tensor_list])
    max_len = max(length_list)
    padded = [pad(t, max_len) for t in tensor_list]
    packed = torch.stack(padded, 0)
    return packed, length_list
