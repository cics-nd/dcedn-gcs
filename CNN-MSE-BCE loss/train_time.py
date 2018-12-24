"""
Convolutional Encoder-Decoder Networks for Image-to-Image Regression

Surrogate: y = f(x, t)

Output:
    regression:
        pressure
        saturation
    classification:
        binary saturation

Add extra binary saturation channel in the output. The network is asked to do
classification, besides regression task.

May 12, 2018

References:
Mo, S., Zhu, Y., Zabaras, N., Shi, X., & Wu, J. (2018). Deep convolutional encoder-decoder networks for uncertainty quantification
of dynamic multiphase flow in heterogeneous media. arXiv preprint arXiv:1807.00882. (accepted by Water Resources Research)

Author: Shaoxing Mo, E-mail: smo@smail.nju.edu.cn

"""

import torch as th
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from torch.utils.data import Dataset
from dense_ed import DenseEDT, DenseEDTI
from data import load_data, load_data_dynamic, load_tensor
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

args = Parser().parse()  # the network parameters
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# #enters time in latent as an extra coarse feature map with a constant value at all pixels
# # 1-one input channel, i.e. the permeability field
# # 3-three output channels, i.e. the pressure, saturation, and binary saturation fields at one time step
model = DenseEDT(1, 3, blocks=args.blocks, times=args.times,
                 growth_rate=args.growth_rate, drop_rate=args.drop_rate,
                 bn_size=args.bn_size, num_init_features=args.init_features,
                 bottleneck=args.bottleneck, time_channels=args.zt).to(device)
print(model)

# load data
train_data_dir = args.data_dir + '/lhs{}_t{}_expanded.hdf5'.format(args.n_train, len(args.times))
test_data_dir = args.data_dir + '/lhs{}_t{}_expanded.hdf5'.format(args.n_test, len(args.times))
interp_data_dir = args.data_dir + '/lhs{}_t{}.hdf5'.format(50, 3) # data used to plot the snapshots of output at t=100, 150, 200 days
train_loader, train_stats = load_data_dynamic(train_data_dir, args.batch_size)
test_loader, test_stats = load_data_dynamic(test_data_dir, args.test_batch_size)
interp_x, interp_y, interp_times = load_tensor(interp_data_dir)
print('Loaded data!')

optimizer = optim.Adam(model.parameters(), lr=args.lr,
                       weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, 
                              verbose=True)

n_out_pixels_train = len(train_loader.dataset) * train_loader.dataset[0][1][:2].numel()
n_out_pixels_test = len(test_loader.dataset) * test_loader.dataset[0][1][:2].numel()


def test(epoch, plot_intv):
    # compute the surrogate quality metrics and plot predictions
    model.eval()
    loss = 0.
    for batch_idx, (input, target, times) in enumerate(test_loader):
        input, target, times = input.to(device), target[:, :2].to(device), \
                               times.to(device)
        with th.no_grad():
            output = model(input, times)
            loss_regression = F.mse_loss(output[:, :2], target, size_average=False)
        loss += loss_regression.item()

        # plot predictions
        if epoch % plot_intv == 0 and batch_idx == len(test_loader) - 1:
            np.random.seed(1)
            n_samples = 4 # number of samples
            idx = th.randperm(interp_x.size(0))[: n_samples]
            x = interp_x[idx]
            y = interp_y[idx, :2*len(interp_times)]
            print(x.shape)
            print(y.shape)

            for i in range(n_samples):
                # x[i]: (iC, iH, iW), y[i]: (T * oC, oH, oW)
                # prepare input, total time steps
                # (iC, iH, iW) --> (T, iC, iH, iW)
                # i.e. organize the data obtained from one model run to n_t training samples, see Section 3.3 in Mo et al.(2018)
                x_input = x[i].unsqueeze(0).expand(len(interp_times), *x.size()[1:])
                print(x_input.size())
                y_output = model(x_input.to(device), interp_times.to(device))
                print(y_output.size())
                # (T, oC, oH, oW) --> (T * oC, oH, oW)
                y_c1, y_c2 = [], []
                for j, _ in enumerate(interp_times):
                    y_c1.append(y_output[j, 0]) # the predicted pressure field at the j-th time step
                    y_c2.append(y_output[j, 1]) # the predicted saturation field at the j-th time step
                y_output = th.cat((th.stack(y_c1), th.stack(y_c2))).data.cpu() # the predicted output fields at all time steps
                print(y_output.size())
                print(y[i].size())
                # (3 * T * oC, oH, oW)
                samples = th.cat((y[i], y_output, y[i] - y_output)).numpy()
                print(samples.shape)
                fig, axes = plt.subplots(3, len(interp_times) * 2, figsize=(3 * len(interp_times) * 2, 7))
                for j, ax in enumerate(fig.axes):
                    ax.set_aspect('equal')
                    ax.set_axis_off()
                    cax = ax.contourf(samples[j], 55, cmap='jet')
                    cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                            format=ticker.ScalarFormatter(useMathText=True))
                    cbar.formatter.set_powerlimits((0, 0))
                    cbar.ax.yaxis.set_offset_position('left')
                    cbar.update_ticks()
                plt.savefig(args.output_dir + '/epoch_{}_output_{}.png'.format(epoch,idx[i]),
                            bbox_inches='tight',dpi=100)
                plt.close(fig)
                print("epoch {}, done with printing sample output {}".format(epoch, i))

    rmse_test = np.sqrt(loss / n_out_pixels_test)
    r2_score = 1 - loss / test_stats['y_var']
    print("epoch: {}, test r2-score:  {:.4f}".format(epoch, r2_score))
    return r2_score, rmse_test

def cal_R2():
    x_r2, y_r2, t_r2 = load_tensor(test_data_dir)
    y_r2 = y_r2.numpy()
    print('Loaded R2 data!')
    ntimes = len(args.times)
    n_test = args.n_test
    y_sum = np.full( (ntimes,2,50,50), 0.0)

    for i in range(n_test):
        y = y_mc[i * ntimes: (i+1) * ntimes, :2]
        y_sum = y_sum + y
    y_mean= y_sum / n_test

    nominator = 0.0
    denominator = 0.0
    for i in range(n_test):
        print(i)
        x = x_r2[i * ntimes: (i+1) * ntimes]
        t = t_r2[i * ntimes: (i+1) * ntimes]
        y = y_r2[i * ntimes: (i+1) * ntimes, :2]
        x, t = x.to(device), t.to(device)
        model.eval()
        with th.no_grad():
            y_pred = model(x, t)
        y_pred = y_pred.data.cpu().numpy()
        nominator = nominator + ((y - y_pred[:, :2])**2).sum()
        denominator = denominator + ((y - y_mean)**2).sum()
    R2 = 1 - nominator/denominator
    print("R2: {}".format(R2))
    return R2


r2_train, r2_test = [], []
rmse_train, rmse_test = [], []
MSEloss, BCEloss = [], []

exp_dir = args.exp_dir

# Network training
tic = time()
for epoch in range(1, args.n_epochs + 1):
    model.train()
    mse = 0.
    for batch_idx, (input, target, times) in enumerate(train_loader):
        input, target, times = input.to(device), target.to(device), \
                               times.to(device)

        # Stage one of network training using the MSE loss (see Algorithm 1 in Mo et al. (2018))
        model.zero_grad()
        output = model(input, times)
        loss_regression = F.mse_loss(output[:, :2], target[:, :2], size_average=False)
        loss_regression.backward()
        optimizer.step()

        # Stage two of network training using the MES-BCE loss (see Algorithm 1 in Mo et al. (2018))
        model.zero_grad()
        output = model(input, times)
        loss_regression = F.mse_loss(output[:, :2], target[:, :2], size_average=False)
        loss_classification = F.binary_cross_entropy(output[:, [2]], target[:, [2]], size_average=False)
        loss = loss_regression + args.lambda_class * loss_classification
        loss.backward()
        optimizer.step()
        mse += loss_regression.item()

    rmse = np.sqrt(mse / n_out_pixels_train)
    print("epoch: {}".format(epoch))
    if epoch % args.log_interval == 0:
        r2_score = 1 - mse / train_stats['y_var']
        print("epoch: {}, training r2-score: {:.4f}".format(epoch, r2_score))
        print(loss_regression.item(), loss_classification.item())
        r2_train.append(r2_score)
        rmse_train.append(rmse)
        # compute the test R2 and RMSE, and plot temporary predictions
        r2_t, rmse_t = test(epoch, plot_intv=args.plot_interval)
        r2_test.append(r2_t)
        rmse_test.append(rmse_t)
        MSEloss.append(loss_regression.item())
        BCEloss.append(loss_classification.item())

    scheduler.step(rmse)

    # save model
    if epoch == args.n_epochs:
        th.save(model.state_dict(), args.exp_dir + "/model_epoch{}.pth".format(epoch))

tic2 = time()
print("Done training {} epochs with {} data using {} seconds".format(args.n_epochs, args.n_train, tic2 - tic))

# plot the rmse, r2-score curve
x = np.arange(args.log_interval, args.n_epochs + args.log_interval,args.log_interval)
plt.figure()
plt.plot(x, r2_train, label="train: {:.3f}".format(np.mean(r2_train[-5: -1])))
plt.plot(x, r2_test, label="test: {:.3f}".format(np.mean(r2_test[-5: -1])))
plt.xlabel('Epoch')
plt.ylabel(r'$R^2$-score')
plt.legend(loc='lower right')
plt.savefig(exp_dir + "/r2.pdf", format='pdf', dpi=500)
plt.close()
np.savetxt(exp_dir + "/r2_train.txt", r2_train)
np.savetxt(exp_dir + "/r2_test.txt", r2_test)

plt.figure()
plt.plot(x, rmse_train, label="train: {:.3f}".format(np.mean(rmse_train[-5: -1])))
plt.plot(x, rmse_test, label="test: {:.3f}".format(np.mean(rmse_test[-5: -1])))
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend(loc='upper right')
plt.savefig(exp_dir + "/rmse.pdf", dpi=500)
plt.close()
np.savetxt(exp_dir + "/rmse_train.txt", rmse_train)
np.savetxt(exp_dir + "/rmse_test.txt", rmse_test)

# save args and time taken
args.times = args.times.tolist()
args_dict = {}
for arg in vars(args):
    args_dict[arg] = getattr(args, arg)
args_dict['time'] = tic2 - tic
n_params, n_layers = model._num_parameters_convlayers()
args_dict['num_layers'] = n_layers
args_dict['num_params'] = n_params
with open(exp_dir + "/args.txt", 'w') as file:
    file.write(json.dumps(args_dict))

# calculate R2
R2_test_self = []
R2_test_s = cal_R2()
R2_test_self.append(R2_test_s)
np.savetxt(exp_dir + "/R2_test_self.txt", R2_test_self)

np.savetxt(exp_dir + "/MSEloss.txt", MSEloss)
np.savetxt(exp_dir + "/BCEloss.txt", BCEloss)
