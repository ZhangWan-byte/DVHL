# modified from https://github.com/apple/ml-dab/blob/main/models/dab.py
# 
# For licensing see accompanying LICENSE.txt file.
# Copyright (C) 2019-2020 Apple Inc. All Rights Reserved.
#
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import scagnostics


class DAB(nn.Module):
    def __init__(self, approximator, hard_layer, device=torch.device('cuda')):
        """ DAB layer simply accepts an approximator model, a hard layer
            and adds syntatic sugar to return the hard output while caching
            the soft version. It also adds a helper fn loss_function() to
            return the DAB loss.

        :param approximator: the approximator nn.Module
        :param hard_layer: the hard layer nn.Module
        :returns: DAB Object
        :rtype: nn.Module

        """
        super(DAB, self).__init__()
        self.device = device
        self.loss_fn = F.mse_loss
        self.hard_layer = hard_layer.apply
        self.approximator = approximator.to(device)

    def loss_function(self):
        """ Simple helper to return the cached loss

        :returns: loss reduced across feature dimension
        :rtype: torch.Tensor

        """
        assert self.true_output.shape[0] == self.approximator_output.shape[0], "batch mismatch"
        batch_size = self.true_output.shape[0]
        return torch.sum(self.loss_fn(self.approximator_output.view(batch_size, -1),
                                      self.true_output.view(batch_size, -1),
                                      reduction='mean'), dim=-1)

    def forward(self, x, **kwargs):
        """ DAB layer simply caches the true and approximator outputs
            and returns the hard output.

        :param x: the input to the DAB / hard fn
        :returns: hard output
        :rtype: torch.Tensor

        """
        self.approximator_output = self.approximator(x, **kwargs)
        self.true_output = self.hard_layer(x, self.approximator_output, self.device)

        # sanity check and return
        assert self.approximator_output.shape == self.true_output.shape, \
            "proxy output {} doesn't match size of hard output [{}]".format(
                self.approximator_output.shape, self.true_output.shape
            )

        return self.true_output


class BaseHardFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, soft_y, hard_fn, *args):
        """ Runs the hard function for forward, cache the output and returns.
            All hard functions should inherit from this, it implements the autograd override.

        :param ctx: pytorch context, automatically passed in.
        :param x: input tensor.
        :param soft_y: forward pass output (logits) of DAB approximator network.
        :param hard_fn: to be passed in from derived class.
        :param args: list of args to pass to hard function.
        :returns: hard_fn(tensor), backward pass using DAB.
        :rtype: torch.Tensor

        """
        hard = hard_fn(x, *args)
        saveable_args = list([a for a in args if isinstance(a, torch.Tensor)])
        ctx.save_for_backward(x, soft_y, *saveable_args)
        return hard

    @staticmethod
    def _hard_fn(x, *args):
        raise NotImplementedError("implement _hard_fn in derived class")

    @staticmethod
    def backward(ctx, grad_out):
        """ Returns DAB derivative.

        :param ctx: pytorch context, automatically passed in.
        :param grad_out: grads coming into layer
        :returns: dab_grad(tensor)
        :rtype: torch.Tensor

        """
        x, soft_y, *args = ctx.saved_tensors
        # print(x.shape, x.requires_grad)
        # print(soft_y.shape, soft_y.requires_grad)
        with torch.enable_grad():
            grad = torch.autograd.grad(outputs=soft_y, inputs=x,
                                       grad_outputs=grad_out,
                                       # allow_unused=True,
                                       retain_graph=True)
            return grad[0], None, None, None


class ScagEstimator(nn.Module):
    def __init__(self, size_in=70000, size_hidden=100, size_out=9):
        super(ScagEstimator, self).__init__()

        self.layerx = nn.Sequential(
            nn.Linear(size_in, size_hidden), 
            nn.ReLU(), 
            nn.Linear(size_hidden, size_hidden), 
            nn.ReLU(), 
            nn.Linear(size_hidden, size_out), 
            nn.ReLU()
        )

        self.layery = nn.Sequential(
            nn.Linear(size_in, size_hidden), 
            nn.ReLU(), 
            nn.Linear(size_hidden, size_hidden), 
            nn.ReLU(), 
            nn.Linear(size_hidden, size_out), 
            nn.ReLU()
        )

    def forward(self, z):
        x = self.layerx(z[:, 0].view(1,-1))
        y = self.layery(z[:, 1].view(1,-1))
        out = x+y
        return out.view(1,-1)


class ScagModule(BaseHardFn):
    @staticmethod
    def _hard_fn(x, *args):
        """calculate metric values

        :param x: (N_low, 2) - dimensionality reduction result z
        """

        # scagnostics
        all_scags = scagnostics.compute(x[:, 0], x[:, 1])

        result_tensor = torch.tensor([list(all_scags.values())]).view(1,-1)

        return result_tensor.to(args[0])

    @staticmethod
    def forward(ctx, x, soft_y, *args):
        return BaseHardFn.forward(ctx, x, soft_y, ScagModule._hard_fn, *args)


class SignumWithMargin(BaseHardFn):
    @staticmethod
    def _hard_fn(x, *args):
        """ x[x < -eps] = -1
            x[x > +eps] = 1
            else x = 0

        :param x: input tensor
        :param args: list of args with 0th element being eps
        :returns: signum(tensor)
        :rtype: torch.Tensor

        """
        eps = args[0] if len(args) > 0 else 0.5
        sig = torch.zeros_like(x)
        sig[x < -eps] = -1
        sig[x > eps] = 1
        return sig

    @staticmethod
    def forward(ctx, x, soft_y, *args):
        return BaseHardFn.forward(ctx, x, soft_y, SignumWithMargin._hard_fn, *args)