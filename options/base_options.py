import argparse, os
from utils import util


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # misc
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--seed', type=int, default=2018, help='random seed of the experiment')
        self.parser.add_argument('--gpu_count', type=int, default=1, help='gpu count: e.g. 1')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--load_epoch', type=str, default='epoch0', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--scenes_to_visualize', type=int, default=10000000)

        # dataset
        self.parser.add_argument('--dataset_mode', type=str, default='sample_per_vehicle', help='chooses how datasets are loaded. [sample_per_vehicle]')
        self.parser.add_argument('--train_root', type=str, default='', help='path to train images (should have subfolders accident, nonaccident')
        self.parser.add_argument('--valid_root', type=str, default='',help='path to valid images (should have subfolders accident, nonaccident)')
        self.parser.add_argument('--test_root', type=str, default='', help='path to test images')
        self.parser.add_argument('--n_test_splits', type=int, default=-1, help='number of test splits')
        self.parser.add_argument('--train_frames_per_scene', type=int, default=20)
        self.parser.add_argument('--valid_frames_per_scene', type=int, default=20)
        self.parser.add_argument('--test_frames_per_scene', type=int, default=20)
        self.parser.add_argument('--shuffle_by_which', type=str, default='frame', help='unit to shuffle the dataset by [sample | scene | frame]')
        self.parser.add_argument('--frames_per_sample', type=int, default=3)
        self.parser.add_argument('--no_shuffle_per_epoch', action='store_true')
        self.parser.add_argument('--label_method', type=str, default='rule_based')
        self.parser.add_argument('--motion_model', type=str, default='ctra')
        self.parser.add_argument('--ttc_threshold', type=float, default=1.0)
        self.parser.add_argument('--train_dataset_proportion', type=float, default=1.0)

        # model
        self.parser.add_argument('--model', type=str, default='per_vehicle', help='chooses which model to use')
        self.parser.add_argument('--optimizer', type=str, default='adam')
        self.parser.add_argument('--keep_prob', type=float, default=0.5, help='keep probability for dropout')
        self.parser.add_argument('--batch_norm', action='store_true')
        self.parser.add_argument('--feature_extractor', type=str, default='vgg16', help='chooses which feature_extractor to use. alexnet, vgg16, resnet50')
        self.parser.add_argument('--n_rgbs_per_sample', type=int, default=1)
        self.parser.add_argument('--n_bbs_per_sample', type=int, default=1)
        self.parser.add_argument('--no_class_imbalance_weighting', action='store_true')
        self.parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
        self.parser.add_argument('--initializer', type=str,  default='he', help='which initializer')
        self.parser.add_argument('--init_bias_value', type=float, default=0.0, help='inital bias value')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        args = vars(self.opt)

        if self.opt.dataset_mode == 'sample_per_vehicle':
            self.opt.frames_per_sample = max(self.opt.n_rgbs_per_sample, self.opt.n_bbs_per_sample)
            self.opt.input_channel_dim = 3*self.opt.n_rgbs_per_sample + self.opt.n_bbs_per_sample
            self.opt.train_frames_to_remove = int(self.opt.train_frames_per_scene - self.opt.ttc_threshold*10 - self.opt.frames_per_sample + 1)
            self.opt.valid_frames_to_remove = int(self.opt.valid_frames_per_scene - self.opt.ttc_threshold*10 - self.opt.frames_per_sample + 1)
            self.opt.test_frames_to_remove = int(self.opt.test_frames_per_scene - self.opt.ttc_threshold*10 - self.opt.frames_per_sample + 1)
        elif self.opt.dataset_mode == 'sample_per_frame':
            self.opt.input_channel_dim =26
        else:
            print("Unknown dataset mode!")
            raise ValueError

        if self.opt.shuffle_by_which not in ['none', 'sample', 'frame']:
            print("Unknown shuffling method")
            raise ValueError

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        if self.isTrain:
            # save to the disk
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            util.mkdirs(expr_dir)
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
