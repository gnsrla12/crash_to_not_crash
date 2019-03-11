import os
import tensorflow


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain

    def forward(self):
        pass

    def optimize_parameters(self, sess, X_train, Y_train):
        pass

    # helper saving function that can be used by subclasses
    def save(self, epoch):
        pass

    # helper loading function that can be used by subclasses
    def load(self, sess, opt):
        pass

    def update_learning_rate():
        pass
