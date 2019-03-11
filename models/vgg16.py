########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np


class vgg16:
	def __init__(self, features, phase, keep_prob=None, feature_extraction=False, sequence=False, opt=None):
		self.opt = opt

		if self.opt.initializer == "xavier":
			self.initializer = self.xavier_init
		elif self.opt.initializer == "he":
			self.initializer = self.he_init
		else:
			print("Unknown initializer")
			raise ValueError

		# Zero-mean input
		with tf.name_scope('preprocess') as scope:
			rgb_mean = [102.153, 100.801, 105.317]
			rgb_std = [61.305, 59.688, 56.209]
			bb_mean = [4]
			bb_std = [18.679]
			mean = rgb_mean*self.opt.n_rgbs_per_sample + bb_mean*self.opt.n_bbs_per_sample
			mean = tf.constant(mean, dtype=tf.float32, shape=[1, 1, 1, self.opt.input_channel_dim], name='img_mean')
			std = rgb_std*self.opt.n_rgbs_per_sample + bb_std*self.opt.n_bbs_per_sample
			std = tf.constant(std, dtype=tf.float32, shape=[1, 1, 1, self.opt.input_channel_dim], name='img_std')
			features = (features-mean)/std

		self.conv_layers(features, phase, self.opt.batch_norm, input_channel=opt.input_channel_dim)
		self.fc_layers(keep_prob)

	def xavier_init(self):
		xavier_init =  tf.contrib.layers.xavier_initializer()
		return xavier_init

	def he_init(self):
		he_init =  tf.contrib.layers.variance_scaling_initializer(
				factor=2.0,
				mode='FAN_IN',
				uniform=False,
				dtype=tf.float32
			)
		return he_init

	def conv_layer(self, input, name_scope, kernel_shape, phase, batch_norm, padding='SAME'):

		with tf.name_scope(name_scope) as scope:
			kernel = tf.get_variable(name_scope+'weights', shape=kernel_shape, dtype=tf.float32, initializer=self.initializer())
			biases = tf.get_variable(name_scope+"bias", [kernel_shape[-1]], tf.float32, tf.constant_initializer(self.opt.init_bias_value))
			conv = tf.nn.conv2d(input, kernel, strides=[1, 1, 1, 1], padding=padding)
			out = tf.nn.bias_add(conv, biases)
			if batch_norm:
				param_init = {					
					"beta":tf.zeros_initializer(), 
                  	"gamma":tf.ones_initializer(),
                  	"moving_mean":tf.zeros_initializer(),
                  	"moving_variance":tf.ones_initializer()
                  	}
				out = tf.contrib.layers.batch_norm(out, is_training=phase, scope=name_scope+'bn', param_initializers=param_init, fused=True)
			conv = tf.nn.relu(out, name=scope)
			return conv, kernel, biases

	def conv_layers(self, features, phase, batch_norm, input_channel=3):
		self.conv_parameters = []

		# conv1_1
		self.conv1_1, kernel, biases = self.conv_layer(features, 'conv1_1', [3, 3, input_channel, 64], phase, batch_norm)
		self.conv_parameters += [kernel, biases]
		# conv1_2
		self.conv1_2, kernel, biases = self.conv_layer(self.conv1_1, 'conv1_2', [3, 3, 64, 64], phase, batch_norm)
		self.conv_parameters += [kernel, biases]
		# pool1
		self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

		# conv2_1
		self.conv2_1, kernel, biases = self.conv_layer(self.pool1, 'conv2_1', [3, 3, 64, 128], phase, batch_norm)
		self.conv_parameters += [kernel, biases]
		# conv2_2
		self.conv2_2, kernel, biases = self.conv_layer(self.conv2_1, 'conv2_2', [3, 3, 128, 128], phase, batch_norm)
		self.conv_parameters += [kernel, biases]
		# pool2
		self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

		# conv3_1
		self.conv3_1, kernel, biases = self.conv_layer(self.pool2, 'conv3_1', [3, 3, 128, 256], phase, batch_norm)
		self.conv_parameters += [kernel, biases]
		# conv3_2
		self.conv3_2, kernel, biases = self.conv_layer(self.conv3_1, 'conv3_2', [3, 3, 256, 256], phase, batch_norm)
		self.conv_parameters += [kernel, biases]
		# conv3_3
		self.conv3_3, kernel, biases = self.conv_layer(self.conv3_2, 'conv3_3', [3, 3, 256, 256], phase, batch_norm)
		self.conv_parameters += [kernel, biases]
		# pool3
		self.pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

		# conv4_1
		self.conv4_1, kernel, biases = self.conv_layer(self.pool3, 'conv4_1', [3, 3, 256, 512], phase, batch_norm)
		self.conv_parameters += [kernel, biases]
		# conv4_2
		self.conv4_2, kernel, biases = self.conv_layer(self.conv4_1, 'conv4_2', [3, 3, 512, 512], phase, batch_norm)
		self.conv_parameters += [kernel, biases]
		# conv4_3
		self.conv4_3, kernel, biases = self.conv_layer(self.conv4_2, 'conv4_3', [3, 3, 512, 512], phase, batch_norm)
		self.conv_parameters += [kernel, biases]
		# pool4
		self.pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

		# conv5_1
		self.conv5_1, kernel, biases = self.conv_layer(self.pool4, 'conv5_1', [3, 3, 512, 512], phase, batch_norm)
		self.conv_parameters += [kernel, biases]
		# conv5_2
		self.conv5_2, kernel, biases = self.conv_layer(self.conv5_1, 'conv5_2', [3, 3, 512, 512], phase, batch_norm)
		self.conv_parameters += [kernel, biases]
		# conv5_3
		self.conv5_3, kernel, biases = self.conv_layer(self.conv5_2, 'conv5_3', [3, 3, 512, 512], phase, batch_norm)
		self.conv_parameters += [kernel, biases]
		# pool5
		self.pool5 = tf.nn.max_pool(self.conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

		shape = int(np.prod(self.pool5.get_shape()[1:]))
		pool5_flat = tf.reshape(self.pool5, [-1, shape])
		return pool5_flat


	def fc_layers(self, keep_prob):
		self.fc_parameters = []

		# fc1
		with tf.variable_scope('fc1') as scope:
			shape = int(np.prod(self.pool5.get_shape()[1:]))
			fc1w = tf.get_variable('fc_weights', shape=[shape, 100], dtype=tf.float32, 
									initializer=self.initializer())
			fc1b = tf.get_variable('fc_biases', [100], tf.float32, 
									tf.constant_initializer(self.opt.init_bias_value))
			pool5_flat = tf.reshape(self.pool5, [-1, shape])
			self.fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
			self.fc1 = tf.nn.relu(self.fc1l)
			self.fc_parameters += [fc1w, fc1b]

			if self.opt.keep_prob < 1.0:
				self.fc1 = tf.nn.dropout(self.fc1, keep_prob)
			else:
				print("Dropout is not used")

		# fc2
		with tf.variable_scope('fc2') as scope:
			fc2w = tf.get_variable('fc_weights', shape=[100, 2], dtype=tf.float32, 
									initializer=self.initializer())
			fc2b = tf.get_variable('fc_biases', [2], tf.float32, 
									tf.constant_initializer(self.opt.init_bias_value))
			self.logits = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
			self.fc_parameters += [fc2w, fc2b]

			self.probs = tf.nn.softmax(self.logits)

