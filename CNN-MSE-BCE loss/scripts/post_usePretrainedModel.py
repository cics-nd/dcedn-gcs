"""
Post-processing functions for plotting the images used in Mo et al. (2018)

plot_label()       - plot the binarized images of the saturation fields prediction by the forward model and network
predict()          - output the network's predictions for the binary images and the corresponding prediction errors
plot_contour()     - plot the snapshots of the pressure and CO2 saturation fields prediction by the forward model and network
plot_interp_time() - plot the predicted pressure and saturation fields at arbitrary time instances (i.e., interpolation)
plot_pdf()         - plot the PDFs of the pressure and saturation at one single location at different times
plot_pdf_oneT()    - plot the PDFs of the saturation at different locations at one single time step
cal_moment()       - compute the mean and variance fields of the pressure and saturation predicted by the forward model and network

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

args = Parser().parse()
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# #enters time in latent as an extra coarse feature map with a constant value at all pixels
model = DenseEDT(1, 3, blocks=args.blocks, times=args.times,
                 growth_rate=args.growth_rate, drop_rate=args.drop_rate,
                 bn_size=args.bn_size, num_init_features=args.init_features,
                 bottleneck=args.bottleneck, time_channels=args.zt).to(device)
print(model)

## Load pretrained model
model_dir = args.exp_dir
model.load_state_dict(th.load(model_dir + '/model_epoch{}.pth'.format(args.n_epochs)))
print('Loaded model')


def plot_label():
    model.eval()
    n_samples = 50
    idx = th.randperm(interp_x.size(0))[: n_samples]
    idx = [0]
    n_samples = len(idx)
    idx = th.LongTensor(np.random.choice(idx, n_samples, replace=False))
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

        target_idx = np.where(y_target > 0.02, 1, 0) # since the sigmoid activation function is used, thus 0.02 instead of 0.0 is used here 
        prediction_idx = np.where(y_output > 0.02, 1, 0)
        err = target_idx - prediction_idx

        # (3 * T * oC, oH, oW)
        samples = np.vstack((target_idx, prediction_idx, err))
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
                     cbar = plt.colorbar(cax, ax=ax, fraction=0.040, pad=0.04,format=ticker.FuncFormatter(lambda x, pos: "%.1f" % x ))
                else:
                     cbar = plt.colorbar(cax, ax=ax, fraction=0.040, pad=0.04,format=ticker.FuncFormatter(lambda x, pos: "%.2f" % x ))
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
        plt.savefig(args.output_dir + '/label_{}.png'.format(idx[i]),bbox_inches='tight',dpi=390)
        plt.close(fig)

def predict():
    # predict the output images
    interp_x, interp_y, interp_times = load_tensor(args.data_dir +'/lhs50_t3_expanded.hdf5')
    ntimes = 3
    y_DCEDN = np.full( (50,1,50,50), 0.0)
    for i in range(50):
        x = interp_x[i * ntimes: (i+1) * ntimes]
        t = interp_times[i * ntimes: (i+1) * ntimes]
        x, t = x.to(device), t.to(device)
        model.eval()
        y_pred = model(x, t)
        y_pred = y_pred.data.cpu().numpy()
        y_DCEDN[i] = y_pred[ntimes-1,[2]]

    hf = h5py.File(args.output_dir +'/N_{}_SgLabel_DCEDN.hdf5'.format(args.n_train), 'w')
    hf.create_dataset('dataset', data = y_DCEDN, dtype ='f', compression = 'gzip')
    hf.close()

def plot_contour():
    model.eval()
    n_samples = 5
    idx = th.LongTensor(np.random.choice(interp_x.size(0), n_samples, replace=False))
    x = interp_x[idx]
    y = interp_y[idx, :2*len(interp_times)]


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
        print(y_output.size())
        print(y[i].size())
        # (3 * T * oC, oH, oW)
        samples = th.cat((y[i], y_output, y[i] - y_output)).numpy()
        print("Shape of samples: {}".format(samples.shape))

        # use the same colormap limits for the pressure, saturation, and prediction error, respectively, to facilitate the comparison
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
                    cbar = plt.colorbar(cax, ax=ax, fraction=0.040, pad=0.04,format=ticker.FuncFormatter(lambda x, pos: "%.3f" % x ))

        plt.savefig(args.output_dir + '/N_{}_output_{}.png'.format(args.n_train, idx[i]),bbox_inches='tight',dpi=400)
        plt.close(fig)
        print("epoch {}, done with printing sample output {}".format(args.n_epochs, idx[i]))

def plot_interp_time():
    # plot the snapshots of P and Sg at 23 time instances
    model.eval()
    Nt = len(interp_times)

    idx0 = [0] # specify the sample index
    idx = th.LongTensor(np.random.choice(idx0, len(idx0), replace=False))
    x = interp_x[idx]
    y_target = interp_y[idx, :2*len(interp_times)]

    for i in range(len(idx0)):
        # x[i]: (iC, iH, iW), y[i]: (T * oC, oH, oW)
        # prepare input, total time steps
        # (iC, iH, iW) --> (T, iC, iH, iW)
        x_input = x[i].unsqueeze(0).expand(len(interp_times), *x.size()[1:])
        y_output = model(x_input.to(device), interp_times.to(device))
        # (T, oC, oH, oW) --> (T * oC, oH, oW)
        y_c1, y_c2 = [], []
        for j, _ in enumerate(interp_times):
            y_c1.append(y_output[j, 0])
            y_c2.append(y_output[j, 1])
        y_output = th.cat((th.stack(y_c1), th.stack(y_c2))).data.cpu()
        # (3 * T * oC, oH, oW)
        y = y_target[i].numpy()
        y_output = y_output.numpy()
        err = y - y_output

        # use the same colormap limits for the pressure, saturation, and prediction error, respectively, to facilitate the comparison
        p = y[:Nt]
        Sg = y[Nt:2*Nt]
        p_hat = y_output[:Nt]
        Sg_hat = y_output[Nt:2*Nt]
        err_p = err[:Nt]
        err_Sg = err[Nt:2*Nt]
        print("p shape: {}".format(err_p.shape))
        cbar_pmax = np.max(np.vstack((p,p_hat)))
        cbar_Sgmax = np.max(np.vstack((Sg,Sg_hat)))
        cbar_err_p = np.max(np.abs(err_p))
        cbar_err_Sg = np.max(np.abs(err_Sg))

        flag = 0 # 0 for the former part, 1 for the later part of Figure 10 in Mo et al.(2018)
        if flag == 0:
            Ncolume = 12
            p = p[:Ncolume]
            Sg = Sg[:Ncolume]
            p_hat = p_hat[:Ncolume]
            Sg_hat = Sg_hat[:Ncolume]
            err_p = err_p[:Ncolume]
            err_Sg = err_Sg[:Ncolume]
        else:
            Ncolume = 11
            p = p[Ncolume+1:Nt]
            Sg =Sg[Ncolume+1:Nt]
            p_hat = p_hat[Ncolume+1:Nt]
            Sg_hat = Sg_hat[Ncolume+1:Nt]
            err_p = err_p[Ncolume+1:Nt]
            err_Sg = err_Sg[Ncolume+1:Nt]

        p_data = np.vstack((p, p_hat, err_p))
        Sg_data = np.vstack((Sg, Sg_hat, err_Sg))

        fig = plt.figure(figsize=(2*Ncolume+1, 12))
        axes = []
        outer = gridspec.GridSpec(2, 1, wspace=0.02, hspace=0.02)
        nlevel = 40
        for j in range(2):
            inner = gridspec.GridSpecFromSubplotSpec(3, Ncolume, subplot_spec = outer[j], wspace=0.01, hspace=0.01)
            l = 0
            for k in range(3*Ncolume):
                ax = plt.Subplot(fig, inner[k])
                ax.set_xticks([])
                ax.set_yticks([])
                if j == 0:
                    if k < 2*Ncolume:
                        cbar_max = cbar_pmax
                        cax = ax.contourf(p_data[k], np.arange(0.0 , cbar_max + cbar_max/nlevel , cbar_max/nlevel), cmap='jet')
                        fig.add_subplot(ax)
                    else:
                        cbar_max = cbar_err_p
                        cax = ax.contourf(p_data[k], np.arange(0.0 - cbar_max - cbar_max/nlevel , cbar_max + cbar_max/nlevel , cbar_max/nlevel), cmap='jet')
                        fig.add_subplot(ax)
                else:
                    if k < 2*Ncolume:
                        cbar_max = cbar_Sgmax
                        cax = ax.contourf(Sg_data[k], np.arange(0.0 , cbar_max + cbar_max/nlevel , cbar_max/nlevel), cmap='jet')
                        fig.add_subplot(ax)
                    else:
                        cbar_max = cbar_err_Sg
                        cax = ax.contourf(Sg_data[k], np.arange(0.0 - cbar_max - cbar_max/nlevel , cbar_max + cbar_max/nlevel , cbar_max/nlevel), cmap='jet')
                        fig.add_subplot(ax)

                ylabel_p = (['$P\'$', '$\hat{P}\'$', '$P\'-\hat{P}\'$'])
                ylabel_Sg = (['$Sg$', '$\hat{S}g$', '$Sg-\hat{S}g$'])
                font_size = 28
                if np.mod(k,Ncolume) == 0:
                    if j == 0:
                        ax.set_ylabel(ylabel_p[l], fontsize = font_size)
                        l = 1 + l
                    else:
                        ax.set_ylabel(ylabel_Sg[l], fontsize = font_size)
                        l = 1 + l
                if j == 1 and k >= 2*Ncolume:
                    if flag:
                        t = (k-2*Ncolume)*5+160
                        if k == 3*Ncolume-2:
                            t = 225
                        if k == 3*Ncolume-1:
                            t = 250
                        ax.set_xlabel("{} d".format(t), fontsize = font_size)
                    else:
                        ax.set_xlabel("{} d".format((k-2*Ncolume)*5+100), fontsize = font_size)
                ax.spines['left'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                if np.mod(k+1,Ncolume) == 0 and flag:
                    cbar = plt.colorbar(cax, ax=ax, fraction=0.040, pad=0.04,
                                     format=ticker.FuncFormatter(lambda x, pos: "%.2f" % x ))
                    tick_locator = ticker.MaxNLocator(nbins=5)
                    cbar.locator = tick_locator
                    cbar.update_ticks()
                    cbar.ax.tick_params(labelsize=font_size-5)
        if flag:
            plt.savefig(args.output_dir + '/interp_output_{}_1.png'.format(idx[i]),
                        bbox_inches='tight',dpi=330)
        else:
            plt.savefig(args.output_dir + '/interp_output_{}_0.png'.format(idx[i]),
                        bbox_inches='tight',dpi=330)
        plt.close(fig)
        print("done with printing sample output {}".format(idx[i]))

def plot_pdf():
    idx = [11,12,13,14] # specify the location index, idx\in[0,49]
    idy = 24
    ntimes = 3
    y_pred_idxy = np.full( (n_test,ntimes,2,len(idx)), 0.0)
    y_idxy = np.full( (n_test,ntimes,2,len(idx)), 0.0)
    for i in range(n_test):
        x = x_mc[i * ntimes: (i+1) * ntimes]
        t = t_mc[i * ntimes: (i+1) * ntimes]
        y = y_mc[i * ntimes: (i+1) * ntimes]
        x, t = x.to(device), t.to(device)
        model.eval()
        y_pred = model(x, t)
        y_pred = y_pred.data.cpu().numpy()
        y_pred_idxy[i] = y_pred[:,:2,idy,idx] # the network predicted outputs at the specified location
        y_idxy[i] = y[:,:2,idy,idx]           # the simulation outputs at the specified location

    for i in range(len(idx)):
        fig, axes = plt.subplots(2, ntimes, figsize=(7, 5))
        for j, axi in enumerate(fig.axes):
            J = j
            nt = 0
            if j >= ntimes:
                J = j - ntimes
                nt = 1
            if j == 2*ntimes-1:
               ax = sns.kdeplot(y_idxy[:,J,nt,i],color='k',label='MC ({:,})'.format(n_test),ax=axi )
               sns.kdeplot(y_idxy[:args.n_train,J,nt,i],color='b',ls="-.",label='MC ({:,})'.format(args.n_train),ax=axi)
               sns.kdeplot(y_pred_idxy[:,J,nt,i],color='r',ls="--",label='Surrogate ({:,})'.format(args.n_train),ax=axi)
            else:
               ax = sns.kdeplot(y_idxy[:,J,nt,i],color='k',ax=axi )
               sns.kdeplot(y_idxy[:args.n_train,J,nt,i],color='b',ls="-.",ax=axi)
               sns.kdeplot(y_pred_idxy[:,J,nt,i],color='r',ls="--",ax=axi)
            if j < ntimes:
               ax.set_xlabel('$P\'$', fontsize=13)
            else:
               ax.set_xlabel('$Sg$', fontsize=13)
            if J == 0:
               ax.set_ylabel('PDF', fontsize=13)
            if j < ntimes:
                ax.set_title('{} days'.format(j*50+100), fontsize=13)
            plt.tight_layout()
        plt.legend(loc = 'lower left', ncol=3,bbox_to_anchor=(-2.7, 2.52), fontsize=13)
        plt.savefig(args.output_dir + "/PDF_x{}_y{}.png".format((idx[i]+1)*10,(idy+1)*10),bbox_inches='tight',dpi=400)
        plt.close(fig)

def plot_pdf_oneT():
    idy = 24
    ntimes = 3
    idt = 2 # the time instance index, 1 for 150 day, and 2 for 200 day
    if idt == 1:
        idx = [16,18,20,22] # specify the location index, idx\in[0,49]
    else:
        idx = [24,26,28,30]

    y_pred_idxy = np.full( (n_test,len(idx)), 0.0)
    y_idxy = np.full( (n_test,len(idx)), 0.0)
    for i in range(n_test):
        x = x_mc[i * ntimes: (i+1) * ntimes]
        t = t_mc[i * ntimes: (i+1) * ntimes]
        y = y_mc[i * ntimes: (i+1) * ntimes]
        x, t = x.to(device), t.to(device)
        model.eval()
        y_pred = model(x, t)
        y_pred = y_pred.data.cpu().numpy()
        y_pred_idxy[i] = y_pred[idt,1,idy,idx]
        y_idxy[i] = y[idt,1,idy,idx]
        print(i)

    fig, axes = plt.subplots(1, len(idx), figsize=(8.5, 2.5))
    for j, axi in enumerate(fig.axes):
        if j == len(idx)-1:
           ax = sns.kdeplot(y_idxy[:,j],color='k',label='MC ({:,})'.format(n_test),ax=axi )
           sns.kdeplot(y_idxy[:args.n_train,j],color='b',ls="-.",label='MC ({:,})'.format(args.n_train),ax=axi)
           sns.kdeplot(y_pred_idxy[:,j],color='r',ls="--",label='Surrogate ({:,})'.format(args.n_train),ax=axi)
        else:
           ax = sns.kdeplot(y_idxy[:,j],color='k',ax=axi )
           sns.kdeplot(y_idxy[:args.n_train,j],color='b',ls="-.",ax=axi)
           sns.kdeplot(y_pred_idxy[:,j],color='r',ls="--",ax=axi)
        ax.set_xlabel('$Sg$')#, fontsize=13)
        if j == 0:
           ax.set_ylabel('PDF')#, fontsize=13)
        ax.set_title('location ({},{})'.format((idx[j]+1)*10,(idy+1)*10))#, fontsize=13)
        plt.tight_layout()
    plt.legend(loc = 'lower left', ncol=3,bbox_to_anchor=(-3.2, 1.13) )
    plt.savefig(args.output_dir + "/PDF_y{}_t{}.png".format((idy+1)*10,100+idt*50),bbox_inches='tight',dpi=400)
    plt.close(fig)

def cal_moment():
    ntimes = 3 # compute the output mean and variance fields at t = 100, 150, and 200 days
    y_sum_MC = np.full( (ntimes,2,50,50), 0.0)
    y_sum_DCEDN = np.full( (ntimes,2,50,50), 0.0)
    for i in range(n_test):
        x = x_mc[i * ntimes: (i+1) * ntimes]
        t = t_mc[i * ntimes: (i+1) * ntimes]
        y = y_mc[i * ntimes: (i+1) * ntimes]
        x, t = x.to(device), t.to(device)
        model.eval()
        y_pred = model(x, t)
        y_pred = y_pred.data.cpu().numpy()
        y_pred = y_pred[:,:2]
        y = y[:,:2]
        y_sum_MC = y_sum_MC + y
        y_sum_DCEDN = y_sum_DCEDN + y_pred
        print(i)
    y_mean_MC = y_sum_MC / n_test
    y_mean_DCEDN = y_sum_DCEDN / n_test
    hf = h5py.File('Moment_mean_MC.hdf5', 'w')
    hf.create_dataset('dataset', data = y_mean_MC, dtype ='f', compression = 'gzip')
    hf.close()
    hf = h5py.File('Moment_mean_DCEDN.hdf5', 'w')
    hf.create_dataset('dataset', data = y_mean_DCEDN, dtype ='f', compression = 'gzip')
    hf.close()

    y_sum_MC = np.full( (ntimes,2,50,50), 0.0)
    y_sum_DCEDN = np.full( (ntimes,2,50,50), 0.0)
    for i in range(n_test):
        x = x_mc[i * ntimes: (i+1) * ntimes]
        t = t_mc[i * ntimes: (i+1) * ntimes]
        y = y_mc[i * ntimes: (i+1) * ntimes]
        x, t = x.to(device), t.to(device)
        model.eval()
        y_pred = model(x, t)
        y_pred = y_pred.data.cpu().numpy()
        y_pred = y_pred[:,:2]
        y = y[:,:2]
        y_sum_MC = y_sum_MC + (y - y_mean_MC)** 2
        y_sum_DCEDN = y_sum_DCEDN + (y_pred - y_mean_DCEDN)**2
        print(i)
    y_std_MC = y_sum_MC / n_test
    y_std_DCEDN = y_sum_DCEDN / n_test
    hf = h5py.File('Moment_std_MC.hdf5', 'w')
    hf.create_dataset('dataset', data = y_std_MC, dtype ='f', compression = 'gzip')
    hf.close()
    hf = h5py.File('Moment_std_DCEDN.hdf5', 'w')
    hf.create_dataset('dataset', data = y_std_DCEDN, dtype ='f', compression = 'gzip')
    hf.close()


####------- plot snapshots of the outputs -------------
interp_data_dir = args.data_dir + '/lhs{}_t{}.hdf5'.format(50, 3)
interp_x, interp_y, interp_times = load_tensor(interp_data_dir)
plot_label()    # plot the binarized images
plot_contour()  # plot the snapshots of the pressure and CO2 saturation fields


# #####---output the network's predictions for the binary images and the corresponding prediction errors
# predict()


# ####------- plot the predicted pressure and saturation fields at arbitrary time instances-------------
# interp_data_dir = args.data_dir + '/lhs{}_t{}.hdf5'.format(20, 23)
# interp_x, interp_y, interp_times = load_tensor(interp_data_dir)
# plot_interp_time()


# ####--Uncertainty quantification-###
# hdf5_dir_expanded = args.data_dir + '/lhs{}_t{}_expanded.hdf5'.format(20000, 3)
# x_mc, y_mc, t_mc = load_tensor(hdf5_dir_expanded)
# y_mc = y_mc.numpy()
# plot_pdf()      # PDFs of the pressure and saturation at one single location at different times
# plot_pdf_oneT() # PDFs of the saturation at different locations at one single time step
# cal_moment()    # compute the mean and variance fields
