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
    def __init__(self, seed=None, fan_in=None):
        self.seed = seed
        self.fan_in = fan_in

    def __call__(self, shape, dtype=None):
        fan_in = self.fan_in if self.fan_in else shape[:-1]
        bound = 1/np.sqrt(np.prod(fan_in))
        return keras.random.uniform(shape, -bound, bound, dtype, self.seed)

    def get_config(self):  # To support serialization
      return {'seed': self.seed}


import math

from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend import random
from keras.src.initializers.initializer import Initializer
from keras.src.saving import serialization_lib


def build_and_train(x, y, inner_units, mode, epochs):
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
                    kernel_initializer=TorchLikeInitializer(fan_in=1),
                    bias_initializer=TorchLikeInitializer(fan_in=1))(tmp)
        tmp = Dense(units=1, activation='linear',
                    kernel_initializer=TorchLikeInitializer(fan_in=inner_units),
                    bias_initializer=TorchLikeInitializer(fan_in=inner_units))(tmp)
    elif mode.endswith('he'):
        tmp = Dense(units=inner_units, activation='relu',
                    kernel_initializer=initializers.HeUniform(),
                    bias_initializer=initializers.HeUniform())(tmp)
        tmp = Dense(units=1, activation='linear',
                    kernel_initializer=initializers.HeUniform(),
                    bias_initializer=initializers.HeUniform())(tmp)
    elif mode.endswith('lecun'):
        tmp = Dense(units=inner_units, activation='relu',
                    kernel_initializer=initializers.LecunUniform(),
                    bias_initializer=initializers.LecunUniform())(tmp)
        tmp = Dense(units=1, activation='linear',
                    kernel_initializer=initializers.LecunUniform(),
                    bias_initializer=initializers.LecunUniform())(tmp)
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


def build_and_train_torch(x, y, inner_units, epochs):
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

def run_test(mode, n_units, epochs, plot_history=False):
    # Generate the data
    x, y, x_gt, y_gt, x_vl, y_vl = generate_data()

    # Preprocess data
    x_s, y_s, x_gt_s, y_gt_s, x_vl_s, y_vl_s, scaler_x, scaler_y = preprocess_data(x, y, x_gt, y_gt, x_vl, y_vl)

    # Build and train a model
    if mode.startswith('keras'):
        mdl, history = build_and_train(x_s, y_s, n_units, mode, epochs)
    else:
        mdl = build_and_train_torch(x_s, y_s, n_units, epochs)

    # Build predictions
    if mode.startswith('keras'):
        y_pred = mdl.predict(x_s, verbose=0)
    else:
        with torch.no_grad():
            y_pred = mdl(torch.from_numpy(x_s)).numpy()

    # Compute the MSE
    mse_val = np.mean(np.square(y_pred - y_s))
    print(f'MSE ({mode}, {n_units} hidden neurons): {mse_val}')

    # Plot
    main_plot(data=(x_s, y_s),
              data_val=(x_vl_s, y_vl_s),
              pred=(x_s, y_pred),
              ground_truth=(x_gt_s, y_gt_s),
              title=f'Trained model ({mode})')
    plt.savefig(f'main_plot ({mode}).pdf')
    plt.close()

    if mode.startswith('keras') and plot_history:
        loss_plot(history, title=f'Loss curve ({mode})')
        plt.savefig(f'loss_plot ({mode}).pdf')
        plt.close()

    return mse_val

def stat_test(mode, n_units, epochs=1000, n_trials=10):
    res = []
    for _ in range(n_trials):
        val = run_test(mode=mode, n_units=n_units, epochs=epochs)
        res.append(val)
    return [mode, n_units, np.mean(res), np.std(res)]

if __name__ == "__main__":
    header = ['mode', 'n_units', 'MSE (mean)', 'MSE (std)']
    res = []
    for n_units in [16, 32, 64]:
        res.append(stat_test(mode='keras-tf-default', n_units=n_units))
        res.append(stat_test(mode='keras-tf-custom', n_units=n_units))
        res.append(stat_test(mode='keras-tf-he', n_units=n_units))
        res.append(stat_test(mode='keras-tf-lecun', n_units=n_units))
        res.append(stat_test(mode='keras-torch-default', n_units=n_units))
        res.append(stat_test(mode='keras-torch-custom', n_units=n_units))
        res.append(stat_test(mode='keras-torch-he', n_units=n_units))
        res.append(stat_test(mode='keras-torch-lecun', n_units=n_units))

    with open('results.csv', 'w') as fp:
        fp.write(','.join(header) + '\n')
        for row in res:
            fp.write(','.join(str(v) for v in row) + '\n')

    # run_test(mode='native-torch-default')
