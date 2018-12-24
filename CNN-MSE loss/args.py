import argparse
import os
import numpy as np
import time


class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='Dense Convolutional Encoder-Decoder Network')
        # default to use cuda
        self.add_argument('--exp-name', type=str, default='co2', help='experiment name')
        self.add_argument('--blocks', type=list, default=(4, 9, 4), help='list of number of layers in each block in decoding net')
        self.add_argument('--growth-rate', type=int, default=24, help='output of each conv')
        self.add_argument('--drop-rate', type=float, default=0, help='dropout rate')
        self.add_argument('--bn-size', type=int, default=8, help='bottleneck size: bn_size * growth_rate')
        self.add_argument('--bottleneck', action='store_true', default=False, help='enable bottleneck in the dense blocks')
        self.add_argument('--init-features', type=int, default=48, help='# initial features after the first conv layer')
        self.add_argument('--data-dir', type=str, default="/afs/crc.nd.edu/user/s/smo/co2UQ/DCEDN_time/", help='data directory')
        self.add_argument('--kle-terms', type=int, default=2500, help='num of KLE terms')
        self.add_argument('--n-train', type=int, default=1600, help="number of training data")
        self.add_argument('--n-test', type=int, default=500, help="number of test data")
        self.add_argument('--times', type=list, default=(100,120,140,160,180,200), help="time instances in the simulation output")
        self.add_argument('--zt', type=int, default=1, help='number of latent feature maps for time')
        self.add_argument('--loss-fn', type=str, default='mse', help='loss function: mse, l1, huber, berhu')
        self.add_argument('--n-epochs', type=int, default=200, help='number of epochs to train (default: 200)')
        self.add_argument('--lr', type=float, default=0.001, help='learnign rate')
        self.add_argument('--weight-decay', type=float, default=5e-4, help="weight decay")
        self.add_argument('--batch-size', type=int, default=100, help='input batch size for training (default: 100)')
        self.add_argument('--test-batch-size', type=int, default=100, help='input batch size for testing (default: 100)')
        self.add_argument('--log-interval', type=int, default=2, help='how many epochs to wait before logging training status')
        self.add_argument('--plot-interval', type=int, default=50, help='how many epochs to wait before plotting training status')

    def parse(self):
        args = self.parse_args()

        # 200 is the max simulation time instance
        args.times = np.array(args.times) / 200
        print(args.times)

        date = "May_30_T1_200"
        args.exp_dir = args.data_dir + date + "/kle_{}/ntrn{}_blocks{}_batch{}_epochs{}_lr{}_wd{}_{}_t{}_ti_K{}".\
            format(args.kle_terms, args.n_train, args.blocks,
                args.batch_size, args.n_epochs, args.lr, args.weight_decay, args.loss_fn, len(args.times),args.growth_rate)

        args.output_dir = args.exp_dir + "/predictions"
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        print('------------ Arguments -------------')
        for k, v in sorted(vars(args).items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return args
