from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--plot', action='store_true')
        self.parser.add_argument('--filter_top_n', type=int, default=-1)
        self.isTrain = False
