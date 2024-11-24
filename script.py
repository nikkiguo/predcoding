# pylint: disable=not-callable
# pylint: disable=no-member

import numpy as np
import torch
import os
import mnist_utils
import functions as F
from network import PredictiveCodingNetwork
import argparse


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def main(cf):
    print(f"device [{cf.device}]")
    
    if not cf.dataset or cf.dataset == "mnist":
        print("loading MNIST data...")
        train_set = mnist_utils.get_fashion_mnist_train_set()
        test_set = mnist_utils.get_fashion_mnist_test_set()
    elif cf.dataset == "fashion_mnist":
        print("loading Fashion MNIST data...")
        train_set = mnist_utils.get_fashion_mnist_train_set()
        test_set = mnist_utils.get_fashion_mnist_test_set()


    img_train = mnist_utils.get_imgs(train_set)
    img_test = mnist_utils.get_imgs(test_set)
    label_train = mnist_utils.get_labels(train_set)
    label_test = mnist_utils.get_labels(test_set)
    

    if cf.data_size is not None:
        test_size = cf.data_size // 5
        img_train = img_train[:, 0 : cf.data_size]
        label_train = label_train[:, 0 : cf.data_size]
        img_test = img_test[:, 0:test_size]
        label_test = label_test[:, 0:test_size]

    msg = "img_train {} img_test {} label_train {} label_test {}"
    print(msg.format(img_train.shape, img_test.shape, label_train.shape, label_test.shape))

    print("performing preprocessing...")
    if cf.apply_scaling:
        img_train = mnist_utils.scale_imgs(img_train, cf.img_scale)
        img_test = mnist_utils.scale_imgs(img_test, cf.img_scale)
        label_train = mnist_utils.scale_labels(label_train, cf.label_scale)
        label_test = mnist_utils.scale_labels(label_test, cf.label_scale)

    if cf.apply_inv and cf.act_fn != F.RELU:
        img_train = F.f_inv(img_train, cf.act_fn)
        img_test = F.f_inv(img_test, cf.act_fn)

    model = PredictiveCodingNetwork(cf)

    with torch.no_grad():
        for epoch in range(cf.n_epochs):
            print(f"\nepoch {epoch}")

            img_batches, label_batches = mnist_utils.get_batches(img_train, label_train, cf.batch_size, cf.percent_data_used, cf.subsample_idx)
            print(f"training on {len(img_batches)} batches of size {cf.batch_size}")
            model.train_epoch(img_batches, label_batches, epoch_num=epoch)

            img_batches, label_batches = mnist_utils.get_batches(img_test, label_test, cf.batch_size, cf.percent_data_used, cf.subsample_idx)
            print(f"testing on {len(img_batches)} batches of size {cf.batch_size}")
            accs = model.test_epoch(img_batches, label_batches)
            print(f"average accuracy {np.mean(np.array(accs))}")

            np.random.seed(cf.seed)
            perm = np.random.permutation(img_train.shape[1])
            img_train = img_train[:, perm]
            label_train = label_train[:, perm]

    # Save model state_dict
    filepath = "models/"
    filename = f'net_{cf.act_fn}_{cf.optim}_seed{cf.seed}_samplsize{cf.percent_data_used}_samplidx{cf.subsample_idx}.pth'
    full_path = os.path.join(filepath, filename)
    model.save(full_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument("--data", type=str)
    parser.add_argument("--o", type=str, help="Optimizer name")
    parser.add_argument("--af", type=str, help="Activation function name")

    # Parse the arguments
    args = parser.parse_args()
    
    cf = AttrDict()

    cf.n_epochs = 10
    cf.data_size = None
    cf.batch_size = 128
    cf.seed = 20
    cf.percent_data_used = 0.2
    cf.subsample_idx = 0 # e.g. subsample_idx = 0, 1, 2, 3, 4 when percent_data_used = 0.2

    cf.apply_inv = True
    cf.apply_scaling = True
    cf.label_scale = 0.94
    cf.img_scale = 1.0

    if not args.data: # default is mnist
        cf.dataset = "mnist"
    else:
        cf.dataset = args.data # "mnist" or "fashion_mnist" or "cifar10"
    # elif args.data == "cifar10":
    #     print("Using cifar10 dataset")

    cf.neurons = [784, 128, 128, 128, 10]
        
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

