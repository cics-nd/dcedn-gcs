import numpy as np
import torch as th
import h5py
import sys
from time import time
from torch.utils.data import DataLoader, Dataset, TensorDataset

__all__ = ['load_data', 'load_data_dynamic']

device = th.device("cuda" if th.cuda.is_available() else "cpu")


def load_data_dynamic(data_dir, batch_size):
    """
    Load data in expanded form (x, t)
    """
    with h5py.File(data_dir, 'r') as f:
        x_data = f['input'][()]
        y_data = f['output'][()]
        times = f['times'][()]
        print("input data shape: {}".format(x_data.shape))
        print("output data shape: {}".format(y_data.shape))
        print("times shape: {}".format(times.shape))

    kwargs = {'num_workers': 4,
              'pin_memory': True} if th.cuda.is_available() else {}
    dataset = TensorDataset(th.FloatTensor(x_data), th.FloatTensor(y_data),
                            th.FloatTensor(times))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)

    y_data_mean = np.mean(y_data[:, :2], 0)
    y_data_var = np.sum((y_data[:, :2] - y_data_mean) ** 2)
    stats = {}
    stats['y_mean'] = y_data_mean
    stats['y_var'] = y_data_var

    return data_loader, stats


def load_data(data_dir, batch_size, time_steps):
    # not expanded data
    with h5py.File(data_dir, 'r') as f:
        x_data = f['input'][()]
        y_data = f['output'][()]
        print("input data shape: {}".format(x_data.shape))
        print("output data shape: {}".format(y_data.shape))

    kwargs = {'num_workers': 4,
              'pin_memory': True} if th.cuda.is_available() else {}

    dataset = TensorDataset(th.FloatTensor(x_data), th.FloatTensor(y_data))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)

    y_data_mean = np.mean(y_data, 0)
    y_data_var = np.sum((y_data - y_data_mean) ** 2)
    stats = {}
    stats['y_mean'] = y_data_mean
    stats['y_var'] = y_data_var

    return data_loader, stats

def load_tensor(data_dir):
    with h5py.File(data_dir, 'r') as f:
        x_data = f['input'][()]
        y_data = f['output'][()]
        times = f['times'][()]
        print("input data shape: {}".format(x_data.shape))
        print("output data shape: {}".format(y_data.shape))
        print("times shape: {}".format(times.shape))

    return th.FloatTensor(x_data, device=device), th.FloatTensor(y_data), th.FloatTensor(times)


def _combine_x_y_t(data_dir, n_data, times):
    """
    Combines dataset of input, output, and times in a single hdf5 file
    """
    with h5py.File(data_dir + '/input_lhs{}_t{}.hdf5'.format(n_data, len(times)), 'r') as f:
        x_train = np.expand_dims(f['dataset'][()], 1)
    with h5py.File(data_dir + '/output_lhs{}_t{}.hdf5'.format(n_data, len(times)), 'r') as f:
        y_train = f['dataset'][()]

    with h5py.File(data_dir + "/lhs{}_t{}.hdf5".format(n_data, len(times)), 'w') as f:
        input = f.create_dataset(name='input', data=x_train, dtype='f', compression='gzip')
        print(input)
        print(input[-1])
        output = f.create_dataset(name='output', data=y_train, dtype='f', compression='gzip')
        print(output)
        times = f.create_dataset(name='times', data=times, dtype='f')
        print(times[()])


def _expand_x_t(data_dir, n_data, times):
    """
    Expands dataset explicitly as (x, t), inner loop of t
    Converts input data format (N, 1, iH, iW) --> (T * N, 1, iH, iW)
    Converts output data format (N, T * oC, oH, oW) --> (T * N, oC, oH, oW)
    times: (T * N,)
    """
    with h5py.File(data_dir + "/lhs{}_t{}.hdf5".format(n_data, len(times)), 'r') as f:
        x_train = f['input'][()]
        y_train = f['output'][()]
        times = f['times'][()]

        # convert input data format (N, 1, iH, iW) --> (T * N, 1, iH, iW)
        x_train_aug = np.zeros((len(times) * x_train.shape[0], *x_train.shape[1:]))
        times_aug = np.zeros(len(times) * x_train.shape[0])
        # (x, t): loop t first
        for i in range(x_train.shape[0]):
            x_train_aug[i * len(times): (i + 1) * len(times)] = x_train[i]
            times_aug[i * len(times): (i + 1) * len(times)] = times
        print("total input shape: {}".format(x_train_aug.shape))
        print("total times shape: {}".format(times_aug.shape))

        # convert output data format (N, T * oC, oH, oW) --> (T * N, oC, oH, oW)
        y_train_aug = np.zeros((len(times) * y_train.shape[0],
                                y_train.shape[1] // len(times),
                                *y_train.shape[2:]))

        for i in range(y_train.shape[0]):
            y_temp = []
            for j in range(len(times)):
                # T x oC x oH x oW
                indices = [j + len(times) * k for k in range(y_train_aug.shape[1])]
                y_temp.append(y_train[i, indices])
            y_train_aug[i * len(times): (i + 1) * len(times)] = np.stack(y_temp)
        print("total output data shape: {}".format(y_train_aug.shape))

        with h5py.File(data_dir + "/lhs{}_t{}_expanded.hdf5".format(n_data, len(times)),
                       'w') as f:
            input = f.create_dataset(name='input', data=x_train_aug, dtype='f',
                                     compression='gzip')
            print(input.shape)
            print(input[-1])
            output = f.create_dataset(name='output', data=y_train_aug, dtype='f',
                                      compression='gzip')
            print(output.shape)
            print(output[-1])
            times = f.create_dataset(name='times', data=times_aug, dtype='f')
            print(times[()])


if __name__ == '__main__':
    n_data = 500
    epsilon = 0.0
    times = np.array([100,120,140,160,180,200]) / 200
    hdf5_dir = "/afs/crc.nd.edu/user/s/smo/co2UQ/DCEDN_binary/data/epsilon{}".format(epsilon)

    _combine_x_y_t(hdf5_dir, n_data, times)
    _expand_x_t(hdf5_dir, n_data, times)
