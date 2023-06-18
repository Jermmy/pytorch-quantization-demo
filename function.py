from torch.autograd import Function
import torch


class FakeQuantize(Function):

    @staticmethod
    def forward(ctx, x, qparam):
        x = qparam.quantize_tensor(x)
        x = qparam.dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
    
def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    x_ = x.reshape(x.size(0), -1)
    xp = xp.unsqueeze(0)
    fp = fp.unsqueeze(0)
        
    m = (fp[:,1:] - fp[:,:-1]) / (xp[:,1:] - xp[:,:-1])  #slope
    b = fp[:, :-1] - (m.mul(xp[:, :-1]) )

    indicies = torch.sum(torch.ge(x_[:, :, None], xp[:, None, :]), -1) - 1  #torch.ge:  x[i] >= xp[i] ? true: false
    indicies = torch.clamp(indicies, 0, m.shape[-1] - 1)

    line_idx = torch.linspace(0, indicies.shape[0], 1, device=indicies.device).to(torch.long)
    line_idx = line_idx.expand(indicies.shape)
    # idx = torch.cat([line_idx, indicies] , 0)
    out = m[line_idx, indicies].mul(x_) + b[line_idx, indicies]
    out = out.reshape(x.shape)
    return out