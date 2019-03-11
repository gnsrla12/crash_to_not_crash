from os import listdir
from os.path import isfile, join
import numpy as np
import tensorflow as tf
import utils.util as util
from .base_model import BaseModel
import data_label
import time

class per_vehicle_model(BaseModel):
    def name(self):
        return 'Per_Vehicle_Model'

    def initialize(self, opt, sess, pos_weight=1):
        self.opt = opt
        self.sess = sess
        self.isTrain = self.opt.isTrain
        self.save_dir = join(self.opt.checkpoints_dir, self.opt.name)

        # Define placeholder tensors for input features.
        self.labels = tf.placeholder(tf.float32, None)
        self.label_probs = tf.placeholder(tf.float32, None)
        self.frame_labels = tf.placeholder(tf.int64, None)
        self.features = tf.placeholder(tf.float32, (None, 130 , 355, opt.input_channel_dim))
        self.keep_prop = tf.placeholder(tf.float32) # probability to keep units
        self.learning_rate = tf.placeholder(tf.float32)
        self.phase = tf.placeholder(tf.bool)

        # Load feature extractor (AlexNet, VGG16, ResNet50)
        if self.opt.feature_extractor == 'vgg16':
            from models.vgg16 import vgg16
            self.network = vgg16(self.features, self.phase, self.keep_prop, feature_extraction=True, opt=self.opt)
        elif self.opt.feature_extractor == 'resnet50':
            from models.resnet import resnet50
            self.network = resnet50(self.features, self.phase, opt=self.opt)
        else:
            raise ValueError("Feature extractor [%s] not recognized." % self.opt.feature_extractor)

        # Define loss funciton, optimizer prediction and accuracy operations.
        if self.opt.isTrain:
            print("Class positive weighting: ", pos_weight)
            class_weights = tf.constant([1, pos_weight])
            weights = tf.gather(class_weights, tf.cast(self.labels, tf.int64))
            if opt.label_method == "rule_based_prob":
                stacked_label_prob = tf.stack([1-self.label_probs, self.label_probs], 1)
                cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(
                                                    stacked_label_prob, 
                                                    self.network.logits,
                                                    weights=weights))
            else:
                if not self.opt.no_class_imbalance_weighting:
                    cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
                                                        tf.cast(self.labels, tf.int64), 
                                                        self.network.logits, 
                                                        weights))
                else:
                    cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
                                                        tf.cast(self.labels, tf.int64), 
                                                        self.network.logits))

            self.loss_op = cross_entropy

        self.prob = tf.nn.softmax(self.network.logits)
        self.preds = tf.argmax(self.network.logits, 1)

        # Define Optimizer for SGD
        if self.opt.isTrain:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if self.opt.optimizer == 'sgd':
                    self.train_op = tf.train.GradientDescentOptimizer(
                                                    learning_rate=self.learning_rate).minimize(self.loss_op)
                elif self.opt.optimizer == 'adam':
                    self.train_op = tf.train.AdamOptimizer(
                                                    learning_rate=self.learning_rate, 
                                                    beta1=self.opt.beta1, 
                                                    beta2=self.opt.beta2).minimize(self.loss_op)
                elif self.opt.optimizer == 'rmsprop':
                    self.train_op = tf.train.RMSPropOptimizer(
                                                    learning_rate=self.learning_rate).minimize(self.loss_op)
                else:
                    raise ValueError(self.opt.optimizer + " Unknown Optimzer")

        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=100)

        # Initialize Network Parameters
        self.sess.run(self.init_op)

        if self.isTrain:
            if self.opt.continue_train:
                self.load(self.opt.load_epoch)
        else:
            if self.opt.load_epoch != 'epoch0':
                self.load(self.opt.load_epoch)

    def optimize_parameters(self, data, learning_rate):
        _, loss = self.sess.run([self.train_op, self.loss_op],
            feed_dict={
                self.features: data['X'],
                self.labels: data['label'],
                self.label_probs: data['label_prob'],
                self.frame_labels: data['frame_label'],
                self.phase: True,
                self.keep_prop: self.opt.keep_prob,
                self.learning_rate: learning_rate})
        return loss

    def forward(self, X):
        probs = self.sess.run(self.prob,
            feed_dict={
                self.features: X,
                self.phase: False,
                self.keep_prop: 1.0})
        return probs

    def logits(self, data):
        logits = self.sess.run(self.network.logits,
            feed_dict={
                self.features: data['X'],
                self.phase: False,
                self.keep_prop: 1.0})
        return logits

    def loss(self, data):
        loss = self.sess.run(self.loss_op,
            feed_dict={
                self.features: data['X'],
                self.labels: data['label'],
                self.label_probs: data['label_prob'],
                self.phase: False,
                self.keep_prop: 1.0})
        return loss

    def predict(self, X):
        probs = self.sess.run(self.prob,
            feed_dict={
                self.features: X,
                self.phase: False,
                self.keep_prop: 1.0})
        preds = [np.argmax(prob) for prob in probs]
        return preds

    # helper saving function that can be used by subclasses
    def save(self, epoch):
        save_path = self.saver.save(self.sess, join(self.save_dir, "epoch" + str(epoch) +".ckpt"))
        print ("Model saved in file: %s" % save_path)
        print("")

    def save_model_with_name(self, name, epoch, valid_roc, test_roc):
        save_path = self.saver.save(self.sess, join(self.save_dir, name+".ckpt"))
        print ("Best Model saved in file: %s" % save_path)
        print("")

        file = open(join(self.save_dir, name+".txt"), "w")
        file.write("Epoch: "+str(epoch)+\
                    " Valid ROC: "+str("%.4f" % valid_roc)+\
                    " Test ROC: "+str("%.4f" % test_roc))
        file.close()

    # helper loading function that can be used by subclasses
    def load(self, load_epoch):
        model_name = str(load_epoch) + ".ckpt"
        restore_model_path = join(self.opt.checkpoints_dir, self.opt.name, model_name)
        self.saver.restore(self.sess, restore_model_path)
        print("Model Restored from %s \n" % restore_model_path)
