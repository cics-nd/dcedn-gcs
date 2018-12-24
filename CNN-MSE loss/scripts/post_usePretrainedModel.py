"""
Post-processing functions for plotting the images used in Mo et al. (2018)

plot_label()       - plot the binarized images of the saturation fields prediction by the forward model and network
plot_contour()     - plot the snapshots of the pressure and CO2 saturation fields prediction by the forward model and network

References:
Mo, S., Zhu, Y., Zabaras, N., Shi, X., & Wu, J. (2018). Deep convolutional encoder-decoder networks for uncertainty quantification
of dynamic multiphase flow in heterogeneous media. arXiv preprint arXiv:1807.00882. (accepted by Water Resources Research)

Author: Shaoxing Mo, E-mail: smo@smail.nju.edu.cn

"""

import torch as th
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from torch.utils.data import Dataset
from dense_ed import DenseEDT, DenseEDTI
from data import load_data, load_data_dynamic, load_tensor,load_tensor_expanded
from args import Parser
import h5py
import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from time import time
import seaborn as sns
from sklearn.metrics import mean_squared_error
plt.switch_backend('agg')


args = Parser().parse()
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# enters time in latent
model = DenseEDT(1, 2, blocks=args.blocks, times=args.times,
                 growth_rate=args.growth_rate, drop_rate=args.drop_rate,
                 bn_size=args.bn_size, num_init_features=args.init_features,
                 bottleneck=args.bottleneck, time_channels=args.zt).to(device)
print(model)


if args.loss_fn == "mse":
    loss_fn = nn.MSELoss(size_average=False)
elif args.loss_fn == "l1":
    loss_fn = nn.L1Loss()
elif args.loss_fn == 'huber':
    # almost same as
    loss_fn = nn.SmoothL1Loss()


def plot_contour():
    model.eval()

    np.random.seed(1)
    n_samples = 10
    idx = th.LongTensor(np.random.choice(interp_x.size(0), n_samples, replace=False))
    x = interp_x[idx]
    y = interp_y[idx]
    print("Index of data: {}".format(idx))
    print(x.shape)
    print(y.shape)

    for i in range(n_samples):
        # x[i]: (1, 50, 50), y[i]: (3 * 3, 50, 50)
        # prepare input, total time steps
        # (1, 50, 50) --> (T, iC, iH, iW)
        x_input = x[i].unsqueeze(0).expand(len(interp_times), *x.size()[1:])
        print(x_input.size())
        y_output = model(x_input.to(device), interp_times.to(device))
        print(y_output.size())
        # (T, oC, oH, oW) --> (T * oC, oH, oW)
        y_c1, y_c2 = [], []
        for j, _ in enumerate(interp_times):
            y_c1.append(y_output[j, 0])
            y_c2.append(y_output[j, 1])
        y_output = th.cat((th.stack(y_c1), th.stack(y_c2))).data.cpu()
        print(y_output.size())
        print(y[i].size())
        # (3 * T * oC, oH, oW)
        samples = th.cat((y[i], y_output, y[i] - y_output)).numpy()
        print("Shape of samples: {}".format(samples.shape))

        # use the same colormap limit for pressure, saturation, and error, respectively
        p = samples[:len(interp_times)]
        Sg = samples[len(interp_times):2*len(interp_times)]
        p_hat = samples[2*len(interp_times):3*len(interp_times)]
        Sg_hat = samples[3*len(interp_times):4*len(interp_times)]
        err_p = samples[4*len(interp_times):5*len(interp_times)]
        err_Sg = samples[5*len(interp_times):6*len(interp_times)]
        print("p shape: {}".format(err_p.shape))
        p_data = np.vstack((p, p_hat, err_p))
        Sg_data = np.vstack((Sg, Sg_hat, err_Sg))
        cbar_pmax = np.max(np.vstack((p,p_hat)))
        cbar_Sgmax = np.max(np.vstack((Sg,Sg_hat)))
        cbar_err_p = np.max(np.abs(err_p))
        cbar_err_Sg = np.max(np.abs(err_Sg))

        fig = plt.figure(figsize=(4*len(interp_times)+1, 6))
        axes = []
        outer = gridspec.GridSpec(1, 2, wspace=0.12, hspace=0.02)
        nlevel = 55
        for j in range(2):
            inner = gridspec.GridSpecFromSubplotSpec(3, len(interp_times), subplot_spec = outer[j], wspace=0.005, hspace=0.005)
            l = 0
            for k in range(3*len(interp_times)):
                ax = plt.Subplot(fig, inner[k])
                ax.set_xticks([])
                ax.set_yticks([])
                if j == 0:
                    if k < 2*len(interp_times):
                        cbar_max = cbar_pmax
                        cax = ax.contourf(p_data[k], np.arange(0.0 , cbar_max + cbar_max/nlevel, cbar_max/nlevel), cmap='jet')
                        fig.add_subplot(ax)
                    else:
                        cbar_max = cbar_err_p
                        cax = ax.contourf(p_data[k], np.arange(0.0 - cbar_max - cbar_max/nlevel, cbar_max + cbar_max/nlevel, cbar_max/nlevel), cmap='jet')
                        fig.add_subplot(ax)
                else:
                    if k < 2*len(interp_times):
                        cbar_max = cbar_Sgmax
                        cax = ax.contourf(Sg_data[k], np.arange(0.0 , cbar_max + cbar_max/nlevel, cbar_max/nlevel), cmap='jet')
                        fig.add_subplot(ax)
                    else:
                        cbar_max = cbar_err_Sg
                        cax = ax.contourf(Sg_data[k], np.arange(0.0 - cbar_max - 0./nlevel, cbar_max + cbar_max/nlevel, cbar_max/nlevel), cmap='jet')
                        fig.add_subplot(ax)

                ylabel_p = (['$\mathbf{y}$', '$\hat{\mathbf{y}}$', '$\mathbf{y}-\hat{\mathbf{y}}$'])
                if np.mod(k,len(interp_times)) == 0:
                    if j == 0:
                        ax.set_ylabel(ylabel_p[l], fontsize=13)
                        l = 1 + l
                    else:
                        l = 1 + l
                xlabel_p = (['100 days', '150 days', '200 days'])
                if k >= 2*len(interp_times):
                    ax.set_xlabel(xlabel_p[k-2*len(interp_times)], fontsize=13)
                ax.spines['left'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                if np.mod(k+1,len(interp_times)) == 0:
                    cbar = plt.colorbar(cax, ax=ax, fraction=0.040, pad=0.04,
                                     format=ticker.FuncFormatter(lambda x, pos: "%.3f" % x ))

        plt.savefig(args.output_dir + '/N_{}_output_{}_ls50.png'.format(args.n_train, idx[i]),
                    bbox_inches='tight',dpi=400)
        plt.close(fig)


# plot the binarized saturation fields
def plot_label():
    model.eval()
    n_samples = 5
    idx = th.randperm(interp_x.size(0))[: n_samples]
    x = interp_x[idx]
    y = interp_y[idx, len(interp_times):2*len(interp_times)]

    for i in range(n_samples):
        # x[i]: (iC, iH, iW), y[i]: (T * oC, oH, oW)
        # prepare input, total time steps
        # (iC, iH, iW) --> (T, iC, iH, iW)
        x_input = x[i].unsqueeze(0).expand(len(interp_times), *x.size()[1:])
        print(x_input.size())
        y_output = model(x_input.to(device), interp_times.to(device))
        print(y_output.size())
        # (T, oC, oH, oW) --> (T * oC, oH, oW)
        y_c1, y_c2 = [], []
        for j, _ in enumerate(interp_times):
            y_c1.append(y_output[j, 0])
            y_c2.append(y_output[j, 1])
        y_output = th.cat((th.stack(y_c1), th.stack(y_c2))).data.cpu()
        y_output = y_output[len(interp_times):2*len(interp_times)]
        print(y_output.size())
        print(y[i].size())

        y_target = y[i]
        y_target = y_target.data.cpu().numpy()
        y_output = y_output.data.cpu().numpy()

        target_idx = np.where(y_target > 0.02, 1, 0)
        prediction_idx = np.where(y_output > 0.02, 1, 0)
        err = target_idx - prediction_idx

        # (3 * T * oC, oH, oW)
        samples = np.vstack((target_idx, prediction_idx, err))

        print(samples.shape)
        fig, axes = plt.subplots(3, len(interp_times), figsize=(2*len(interp_times)+0.5, 6.))
        plt.subplots_adjust(wspace=0.002, hspace=0.02)
        for j, ax in enumerate(fig.axes):
            ax.set_xticks([])
            ax.set_yticks([])
            if j < 2*len(interp_times):
                cax = ax.imshow(samples[j], cmap='jet', origin='lower')
            else:
                cax = ax.imshow(samples[j], vmin=-1.0, vmax=1.0, cmap='jet', origin='lower')
            if j == 0:
                ax.set_ylabel("$\mathbf{\zeta}(Sg)$", fontsize=13)
            if j == len(interp_times):
                ax.set_ylabel("$\mathbf{\zeta}(\hat{S}g)$", fontsize=13)
            if j == 2*len(interp_times):
                ax.set_ylabel("$\mathbf{\zeta}(Sg)-\mathbf{\zeta}(\hat{S}g)$", fontsize=13)
            xlabel = (['100 days', '150 days', '200 days'])
            if j >= 2*len(interp_times):
                ax.set_xlabel(xlabel[j-2*len(interp_times)], fontsize=13)
            if np.mod(j+1,len(interp_times)) == 0:
                if j < 2*len(interp_times):
                     cbar = plt.colorbar(cax, ax=ax, fraction=0.040, pad=0.04,
                                   format=ticker.FuncFormatter(lambda x, pos: "%.1f" % x ))
                else:
                     cbar = plt.colorbar(cax, ax=ax, fraction=0.040, pad=0.04,
                                   format=ticker.FuncFormatter(lambda x, pos: "%.2f" % x ))
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')

        plt.savefig(args.output_dir + '/label_{}.png'.format(idx[i]),
                    bbox_inches='tight',dpi=390)
        plt.close(fig)


# load data
interp_data_dir = args.data_dir + 'data/lhs{}_t{}.hdf5'.format(50, 3) # data used to plot the results at t=100, 150, and 200 days
interp_x, interp_y, interp_times = load_tensor(interp_data_dir)
print('Loaded data!')

# post-processing using the pretrained model
#load pretrained model
model_dir = args.exp_dir
model.load_state_dict(th.load(model_dir + '/model_epoch200.pth'))
print('Loaded model')
plot_contour()
plot_label()
