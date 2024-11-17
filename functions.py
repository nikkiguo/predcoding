# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import numpy as np

LINEAR = "LINEAR"
TANH = "TANH"
LOGSIG = "LOGSIG"
RELU = "RELU"
LEAKY_RELU = "L_RELU"
L_RELU_SLOPE = 0.01

def f(x, act_fn):
    """ (activation_size, batch_size) """
    if act_fn is LINEAR:
        m = x
    elif act_fn is TANH:
        m = torch.tanh(x)
    elif act_fn is LOGSIG:
        return 1. / (torch.ones_like(x) + torch.exp(-x))
    elif act_fn is RELU:
        m = torch.relu(x)
    elif act_fn is LEAKY_RELU:
        m = torch.nn.functional.leaky_relu(x, negative_slope=L_RELU_SLOPE)
    else:
        raise ValueError(f"{act_fn} not supported")
    return m


def f_deriv(x, act_fn):
    """ (activation_size, batch_size) """
    if act_fn is LINEAR:
        deriv = torch.ones(x.shape)
    elif act_fn is TANH:
        deriv = torch.ones_like(x) - torch.tanh(x) ** 2
    elif act_fn is LOGSIG:
        """ TODO """
        f = 1. / (torch.ones_like(x) + torch.exp(-x))
        deriv = torch.mul(f, (torch.ones_like(x) - f))
    elif act_fn is RELU:
        deriv = torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
    elif act_fn is LEAKY_RELU:
        deriv = torch.where(x > 0, torch.ones_like(x, device=x.device), torch.full_like(x, L_RELU_SLOPE, device=x.device))
    else:
        raise ValueError(f"{act_fn} not supported")
    return deriv


def f_inv(x, act_fn):
    """ (activation_size, batch_size) """
    if act_fn is LINEAR:
        m = x
    elif act_fn is TANH:
        num = np.ones(x.shape) + x
        div = (np.ones(x.shape) - x) + 1e-7
        m = 0.5 * np.log(np.divide(num, div))
    elif act_fn is LOGSIG:
        """ TODO """
        div = (np.ones(x.shape) - x) + 1e-7
        m = np.log(np.divide(x, div) + 1e-7)
    elif act_fn is RELU:
        m = x # not applying inverse (doesn't exist inverse for relu)
    elif act_fn is LEAKY_RELU:
        m = np.where(x >= 0, x, x / L_RELU_SLOPE)
    else:
        raise ValueError(f"{act_fn} not supported")
    return m

