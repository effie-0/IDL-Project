import random
from torch.distributions.multinomial import Multinomial

class SAP(nn.Module):
    """
    The original paper is https://arxiv.org/abs/1803.01442.
    Attributes
    ----------
    self.ratio float : ratio of pruning which can be larger than 1.0.
    self.is_valid bool : if this flag is True, inject SAP.
    """
    def __init__(self, ratio=1.0, is_valid=False):
        """
        Parameters
        ----------
        ratio float : ratio of pruning which can be larger than 1.0.
        is_valid bool : if this flag is True, inject SAP.
        """
        super(SAP, self).__init__()
        self.ratio = ratio
        self.is_valid = is_valid

    def forward(self, inputs):
        """
        If self.training or not self.is_valid, just return inputs.
        If self.is_valid apply SAP to inputs and return the result tensor.
        Parameters
        ----------
        inputs torch.Tensor : input tensor whose shape is [b, c, h, w].
        Returns
        -------
        outputs torch.Tensor : just return inputs or stochastically pruned inputs.
        """
        # print("SAP: ", self.is_valid)
        if self.training or not self.is_valid:
            return inputs
        else:
            # print("IN: ", inputs)
            b, c, h, w = inputs.shape
            inputs_1d = inputs.reshape([b, c * h * w])  # [b, c * h * w]
            outputs = torch.zeros_like(inputs_1d)  # outputs with 0 initilization
            inputs_1d_sum = torch.sum(torch.abs(inputs_1d), dim=-1, keepdim=True)
            inputs_1d_prob = torch.abs(inputs_1d) / inputs_1d_sum

            repeat_num = int(c * h * w * self.ratio)
            idx = Multinomial(repeat_num, inputs_1d_prob).sample()
            outputs[idx.nonzero(as_tuple=True)] = inputs_1d[idx.nonzero(as_tuple=True)]
            outputs = outputs / (1 - (1 - inputs_1d_prob) ** repeat_num + 1e-12)
            outputs = outputs.reshape([b, c, h, w])  # [b, c, h, w]
            # print("OUT: ", outputs)
        return outputs   
