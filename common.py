from copy import deepcopy
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ModelEMA(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def forward(self, input):
        return self.module(input)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.parameters(), model.parameters()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))
            for ema_v, model_v in zip(self.module.buffers(), model.buffers()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(model_v)

    def update_parameters(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dropout=0, T=1.0, only_feat=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.T = T
        self.only_feat = only_feat
        self.standardize = nn.LayerNorm(in_features, elementwise_affine=False)
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.register_buffer('dropout', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        input = self.standardize(input)
        if self.dropout is not None:
            input = self.dropout(input)
        input = F.normalize(input, dim=1)
        if self.only_feat:
            return input
        norm_weight = F.normalize(self.weight, dim=1)
        output = input.matmul(norm_weight.t()) / self.T
        if self.bias is not None:
            output += self.bias
        return output


class Calibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, 1))

    def forward(self, input):
        T = F.relu(self.weight)
        return input / T


class LinearClassifier(nn.Module):
    def __init__(self, in_features, out_features, bias=True, zero_init=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        if not zero_init:
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        output = input.matmul(self.weight.t())
        if self.bias is not None:
            output += self.bias
        return output
