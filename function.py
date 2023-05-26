'''
adapted from https://github.com/fungtion/DANN_py3/blob/master/functions.py
by James Kim
May 25, 2023
'''

from torch.autograd import Function

class GradientReversalLayerF(Function):
    @staticmethod
    def forward(ctx, x, lamda):
        ctx.lamda = lamda
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lamda
        return output, None