import numpy as np
import keras
from keras.layers import Input, Dense
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras import optimizers
from keras import initializers
from keras.initializers import RandomNormal, RandomUniform
import os
import math

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import lightning as L
from torch.utils.data import Dataset


# CONFIGURATION
n_units = 32
epochs = 1000

def generate_data():
    n_tr = 400
    n_vl = 100
    n_gt = 2000
    # Define the input range
    x = np.linspace(-1, 1, n_tr)
    x_vl = np.linspace(-1, 1, n_vl)
    x_gt = np.linspace(-1, 1, n_gt)
    # Compute the output values
    f = lambda x: np.sin(2 * np.pi * x)
    y = f(x)
    y_vl = f(x_vl)
    y_gt = f(x_gt)
    # Apply some noise
    x, y, x_gt, y_gt, x_vl, y_vl = (x.astype('float32'), y.astype('float32'),
                                   x_gt.astype('float32'), y_gt.astype('float32'),
                                   x_vl.astype('float32'), y_vl.astype('float32'))
    return x, y, x_gt, y_gt, x_vl, y_vl


def main_plot(data=None,
         data_val=None,
         pred=None,
         ground_truth=None,
         figure_params={"figsize": (6, 4)},
         title=None):
    fig = plt.figure(**figure_params)
    if data:
        plt.scatter(data[0], data[1], s=1, color='tab:green', label='data')
    if data_val:
        plt.scatter(data_val[0], data_val[1], s=1, color='tab:olive', label='data')
    if pred:
        plt.plot(pred[0], pred[1], color='tab:orange', label='predictions')
    if ground_truth:
        plt.plot(ground_truth[0], ground_truth[1], color='tab:blue', label='ground truth')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.tight_layout()
    return fig


def preprocess_data(x, y, x_gt, y_gt, x_vl, y_vl):
    # Build a scaler
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    # Scale the data
    x_s = scaler_x.fit_transform(x.reshape(-1, 1))
    y_s = scaler_y.fit_transform(y.reshape(-1, 1))
    x_vl = scaler_x.transform(x_vl.reshape(-1, 1))
    y_vl = scaler_y.transform(y_vl.reshape(-1, 1))
    x_gt = scaler_x.transform(x_gt.reshape(-1, 1))
    y_gt = scaler_y.transform(y_gt.reshape(-1, 1))
    return x_s, y_s, x_gt, y_gt, x_vl, y_vl, scaler_x, scaler_y



class TorchLikeInitializer(keras.initializers.Initializer):
    def __init__(self, seed=None):
        self.seed = seed

    def __call__(self, shape, dtype=None):
        bound = 1/np.sqrt(np.prod(shape[:-1]))
        return keras.random.uniform(shape, -bound, bound, dtype, self.seed)

    def get_config(self):  # To support serialization
      return {'seed': self.seed}


import math

from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend import random
from keras.src.initializers.initializer import Initializer
from keras.src.saving import serialization_lib


# class MyRandomInitializer(Initializer):
#     def __init__(self, seed=None):
#         self._init_seed = seed
#         if seed is None:
#             seed = random.make_default_seed()
#         elif isinstance(seed, dict):
#             seed = serialization_lib.deserialize_keras_object(seed)
#         elif not isinstance(seed, (int, random.SeedGenerator)):
#             raise ValueError(
#                 "`seed` argument should be an instance of "
#                 "`keras.random.SeedGenerator()` or an integer. "
#                 f"Received: seed={seed}"
#             )
#         self.seed = seed
#
#     def get_config(self):
#         seed_config = serialization_lib.serialize_keras_object(self._init_seed)
#         return {"seed": seed_config}
#
#
#
# def compute_fans(shape):
#     """Computes the number of input and output units for a weight shape.
#
#     Args:
#         shape: Integer shape tuple.
#
#     Returns:
#         A tuple of integer scalars: `(fan_in, fan_out)`.
#     """
#     shape = tuple(shape)
#     if len(shape) < 1:  # Just to avoid errors for constants.
#         fan_in = fan_out = 1
#     elif len(shape) == 1:
#         fan_in = fan_out = shape[0]
#     elif len(shape) == 2:
#         fan_in = shape[0]
#         fan_out = shape[1]
#     else:
#         # Assuming convolution kernels (2D, 3D, or more).
#         # kernel shape: (..., input_depth, depth)
#         receptive_field_size = 1
#         for dim in shape[:-2]:
#             receptive_field_size *= dim
#         fan_in = shape[-2] * receptive_field_size
#         fan_out = shape[-1] * receptive_field_size
#     return int(fan_in), int(fan_out)
#
#
#
# class MyVarianceScaling(MyRandomInitializer):
#     """Initializer that adapts its scale to the shape of its input tensors.
#
#     With `distribution="truncated_normal" or "untruncated_normal"`, samples are
#     drawn from a truncated/untruncated normal distribution with a mean of zero
#     and a standard deviation (after truncation, if used) `stddev = sqrt(scale /
#     n)`, where `n` is:
#
#     - number of input units in the weight tensor, if `mode="fan_in"`
#     - number of output units, if `mode="fan_out"`
#     - average of the numbers of input and output units, if `mode="fan_avg"`
#
#     With `distribution="uniform"`, samples are drawn from a uniform distribution
#     within `[-limit, limit]`, where `limit = sqrt(3 * scale / n)`.
#
#     Examples:
#
#     >>> # Standalone usage:
#     >>> initializer = VarianceScaling(
#         scale=0.1, mode='fan_in', distribution='uniform')
#     >>> values = initializer(shape=(2, 2))
#
#     >>> # Usage in a Keras layer:
#     >>> initializer = VarianceScaling(
#         scale=0.1, mode='fan_in', distribution='uniform')
#     >>> layer = Dense(3, kernel_initializer=initializer)
#
#     Args:
#         scale: Scaling factor (positive float).
#         mode: One of `"fan_in"`, `"fan_out"`, `"fan_avg"`.
#         distribution: Random distribution to use.
#             One of `"truncated_normal"`, `"untruncated_normal"`, or `"uniform"`.
#         seed: A Python integer or instance of
#             `keras.backend.SeedGenerator`.
#             Used to make the behavior of the initializer
#             deterministic. Note that an initializer seeded with an integer
#             or `None` (unseeded) will produce the same random values
#             across multiple calls. To get different random values
#             across multiple calls, use as seed an instance
#             of `keras.backend.SeedGenerator`.
#     """
#
#     def __init__(
#         self,
#         scale=1.0,
#         mode="fan_in",
#         distribution="truncated_normal",
#         seed=None,
#     ):
#         if scale <= 0.0:
#             raise ValueError(
#                 "Argument `scale` must be positive float. "
#                 f"Received: scale={scale}"
#             )
#         allowed_modes = {"fan_in", "fan_out", "fan_avg"}
#         if mode not in allowed_modes:
#             raise ValueError(
#                 f"Invalid `mode` argument: {mode}. "
#                 f"Please use one of {allowed_modes}"
#             )
#         distribution = distribution.lower()
#         if distribution == "normal":
#             distribution = "truncated_normal"
#         allowed_distributions = {
#             "uniform",
#             "truncated_normal",
#             "untruncated_normal",
#         }
#         if distribution not in allowed_distributions:
#             raise ValueError(
#                 f"Invalid `distribution` argument: {distribution}."
#                 f"Please use one of {allowed_distributions}"
#             )
#         self.scale = scale
#         self.mode = mode
#         self.distribution = distribution
#         super().__init__(seed=seed)
#
#     def __call__(self, shape, dtype=None):
#         scale = self.scale
#         fan_in, fan_out = compute_fans(shape)
#         if self.mode == "fan_in":
#             scale /= max(1.0, fan_in)
#         elif self.mode == "fan_out":
#             scale /= max(1.0, fan_out)
#         else:
#             scale /= max(1.0, (fan_in + fan_out) / 2.0)
#         if self.distribution == "truncated_normal":
#             stddev = math.sqrt(scale) / 0.87962566103423978
#             return random.truncated_normal(
#                 shape, mean=0.0, stddev=stddev, dtype=dtype, seed=self.seed
#             )
#         elif self.distribution == "untruncated_normal":
#             stddev = math.sqrt(scale)
#             return random.normal(
#                 shape, mean=0.0, stddev=stddev, dtype=dtype, seed=self.seed
#             )
#         else:
#             limit = math.sqrt(3.0 * scale)
#             print(f'limit: {limit}')
#             return random.uniform(
#                 shape, minval=-limit, maxval=limit, dtype=dtype, seed=self.seed
#             )
#
#     def get_config(self):
#         base_config = super().get_config()
#         config = {
#             "scale": self.scale,
#             "mode": self.mode,
#             "distribution": self.distribution,
#         }
#         return {**base_config, **config}



def build_and_train(x, y, inner_units, mode):
    # Set the backend
    if mode.startswith('keras-tf'):
        os.environ["KERAS_BACKEND"] = "tensorflow"
    elif mode.startswith('keras-torch'):
        os.environ["KERAS_BACKEND"] = "torch"
    elif mode == 'native-torch':
        pass
    else:
        raise Exception('Invalid backend spefication')

    # Build the model
    tmp = Input(shape=(1,))
    mdl_in = tmp

    if mode.endswith('custom'):
        tmp = Dense(units=inner_units, activation='relu',
                    kernel_initializer=TorchLikeInitializer(),
                    bias_initializer=TorchLikeInitializer())(tmp)
        tmp = Dense(units=1, activation='linear',
                    kernel_initializer=TorchLikeInitializer(),
                    bias_initializer=TorchLikeInitializer())(tmp)
    else:
        tmp = Dense(units=inner_units, activation='relu')(tmp)
        tmp = Dense(units=1, activation='linear')(tmp)

    mdl_out = tmp
    mdl = keras.Model(mdl_in, mdl_out)

    # Train
    opt = optimizers.Adam(learning_rate=0.002)
    mdl.compile(loss="mean_squared_error", optimizer=opt)
    history = mdl.fit(x, y, batch_size=32, epochs=epochs, verbose=0)
    return mdl, history


def build_and_train_torch(x, y, inner_units):
    # Custom dataset object
    class MyDataset(Dataset):
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return self.x[idx, :], self.y[idx]


    # Define a lightning modules
    class LitNN(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.nn = nn.Sequential(nn.Linear(x.shape[1], inner_units), nn.ReLU(), nn.Linear(inner_units, 1))

        def forward(self, x):
            return self.nn(x)

        def training_step(self, batch, batch_idx):
            # training_step defines the train loop.
            x, y = batch
            # x = x.view(x.size(0), -1)
            y_hat = self.nn(x)
            loss = F.mse_loss(y_hat, y)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer

    # Build the dataset loader
    train_loader = DataLoader(MyDataset(x, y))

    # Build the model
    mdl = LitNN()

    # train model
    trainer = L.Trainer(accelerator='cpu', max_epochs=epochs, enable_progress_bar=False)
    trainer.fit(model=mdl, train_dataloaders=train_loader)

    return mdl


def loss_plot(history,
         figure_params={"figsize": (6, 4)},
         title=None):
    fig = plt.figure(**figure_params)
    for name, vals in history.history.items():
        plt.plot(vals, label=name)
    plt.xlabel('epochs')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    return fig

def run_test(mode):
    # Generate the data
    x, y, x_gt, y_gt, x_vl, y_vl = generate_data()
    # Preprocess data
    x_s, y_s, x_gt_s, y_gt_s, x_vl_s, y_vl_s, scaler_x, scaler_y = preprocess_data(x, y, x_gt, y_gt, x_vl, y_vl)
    # Build and train a model
    if mode.startswith('keras'):
        mdl, history = build_and_train(x_s, y_s, n_units, mode)
    else:
        mdl = build_and_train_torch(x_s, y_s, n_units)
    # Build predictions
    if mode.startswith('keras'):
        y_pred = mdl.predict(x_s, verbose=0)
    else:
        with torch.no_grad():
            y_pred = mdl(torch.from_numpy(x_s)).numpy()

    # Manually computed MSE
    mse_val = np.mean(np.square(y_pred - y_s))
    print(f'Final MSE ({mode}): {mse_val}')

    # Plot
    main_plot(data=(x_s, y_s),
              data_val=(x_vl_s, y_vl_s),
              pred=(x_s, y_pred),
              ground_truth=(x_gt_s, y_gt_s),
              title=f'Trained model ({mode})')
    plt.savefig(f'main_plot ({mode}).pdf')

    if mode.startswith('keras'):
        loss_plot(history, title=f'Loss curve ({mode})')
        plt.savefig(f'loss_plot ({mode}).pdf')

if __name__ == "__main__":
    run_test(mode='keras-tf-default')
    run_test(mode='keras-tf-custom')
    run_test(mode='keras-torch-default')
    run_test(mode='keras-torch-custom')
    run_test(mode='native-torch-default')
