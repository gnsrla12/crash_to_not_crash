from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--random_train', action='store_true')
        self.parser.add_argument('--nepoch', type=int, default=6, help='# of epoch')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial convolution layer learning rate for adam')
        self.parser.add_argument('--decay_lr_per', type=float, default=5)
        self.parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='beta 1')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='beta 2')
        
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--init_steps_to_skip_eval', type=int, default=-1, help='number of initial training steps to skip evaluation')
        self.parser.add_argument('--n_train_samples_to_eval', type=int, default=128, help='number of train samples to use for evaluation')
        self.parser.add_argument('--n_valid_samples_to_eval', type=int, default=128, help='number of valid samples to use for evaluation')
        self.parser.add_argument('--logger_mode', type=str, default='simple', help='logger mode')
        self.parser.add_argument('--eval_freq', type=int, default=1000, help='evaluate frequency')
        self.isTrain = True
