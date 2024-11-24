# pylint: disable=not-callable
# pylint: disable=no-member

import numpy as np
import torch
import pickle
import mnist_utils
import functions as F

def load(filepath):
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model

class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def main(cf):
    print(f"device [{cf.device}]")
    print("loading MNIST data...")
    test_set = mnist_utils.get_mnist_test_set()

    img_test = mnist_utils.get_imgs(test_set)
    label_test = mnist_utils.get_labels(test_set)

    if cf.data_size is not None:
        test_size = cf.data_size // 5
        img_test = img_test[:, 0:test_size]
        label_test = label_test[:, 0:test_size]

    msg = "img_test {} label_test {}"
    print(msg.format(img_test.shape, label_test.shape))

    print("performing preprocessing...")
    if cf.apply_scaling:
        img_test = mnist_utils.scale_imgs(img_test, cf.img_scale)
        label_test = mnist_utils.scale_labels(label_test, cf.label_scale)

    if cf.apply_inv and cf.act_fn != F.RELU:
        img_test = F.f_inv(img_test, cf.act_fn)

    model = load('models/net_RELU_ADAM_seed20_samplsize0.2_samplidx1.pth')

    with torch.no_grad():
        for epoch in range(cf.n_epochs):
            print(f"\nepoch {epoch}")

            img_batches, label_batches = mnist_utils.get_batches(img_test, label_test, cf.batch_size, cf.percent_data_used, cf.subsample_idx)
            print(f"testing on {len(img_batches)} batches of size {cf.batch_size}")
            accs = model.test_epoch(img_batches, label_batches)
            print(f"average accuracy {np.mean(np.array(accs))}")



if __name__ == "__main__":
    cf = AttrDict()

    cf.n_epochs = 1
    cf.data_size = None
    cf.batch_size = 128
    cf.seed = 20
    cf.percent_data_used = 0.2
    cf.subsample_idx = 0

    cf.apply_inv = True
    cf.apply_scaling = True
    cf.label_scale = 0.94
    cf.img_scale = 1.0

    cf.neurons = [784, 500, 500, 10]
    cf.n_layers = len(cf.neurons)
    cf.act_fn = F.RELU
    cf.var_out = 1
    cf.vars = torch.ones(cf.n_layers)

    cf.itr_max = 50
    cf.beta = 0.1
    cf.div = 2
    cf.condition = 1e-6
    cf.d_rate = 0

    # optim parameters
    cf.l_rate = 1e-3
    cf.optim = "ADAM"
    cf.eps = 1e-8
    cf.decay_r = 0.9
    cf.beta_1 = 0.9
    cf.beta_2 = 0.999

    cf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(cf)

