import numpy as np
import tensorflow as tf
from scipy.misc import imresize, imread
from os.path import isfile, join, exists, basename, splitext
from os import listdir
from random import shuffle
import threading, pickle, glob, json, math, random, itertools
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial

import data_label
import utils.util as util

def build_scene_list(path, frames_per_scene, frames_to_remove, frame_label):
	scene_list = list(chunks(
							build_frame_list(path, frame_label),
							frames_per_scene, 
							offset=frames_to_remove))
	return scene_list

# Yield successive n-sized chunks from l.
def chunks(list, size, offset=0):
	for i in range(0, len(list), size):
		yield list[i+offset:i + size]

def build_frame_list(path, frame_label):
	jpg_list = []
	print(path)
	if exists(path):
		jpg_list = glob.glob(join(path,"*.jpg"))
		jpg_list = sorted(jpg_list, key=lambda jpg: int(basename(jpg).split(".")[0]))
		print("# of jpg files in path {} : {}".format(path,len(jpg_list)))
	else:
		print("No file found at ", path)
	frame_list = [{"json_path":splitext(jpg)[0]+".json", 
					"img_path":jpg, 
					"frame_label": frame_label} 
					for jpg in jpg_list]
	return frame_list

def get_vehicle_label(bbox, motion_model, ttc_threshold):
	if 'label' in bbox:
		data_driven_label = bbox['label']
		rule_based_label = bbox['label']
		rule_based_prob = None
	else:
		data_driven_label = bbox['syntheticLabel']
		if motion_model == 'ctra': 
			rule_based_label = bbox['adaptedLabel']
			rule_based_prob =  None # TBA
		elif motion_model == 'cca':
			rule_based_label = None # TBA
			rule_based_prob =  None # TBA
		else:
			ValueError("Unknown motion model")


	return data_driven_label, rule_based_label, rule_based_prob


def frame_per_vehicle(frame, vehicle_bbox):
	frame_per_vehicle = {
		'img_path' : frame['img_path'],
		'json_path' : frame['json_path'],
		'bbox' : vehicle_bbox,
	}
	return frame_per_vehicle

def get_bbox(frame_info, hashcode):
	for bbox in frame_info['vehicleInfo']:
		if hashcode == bbox['hashcode']:
			return bbox
	return None

def build_sample_list(scenes, frames_per_sample, label_method, motion_model, ttc_threshold, shuffle_by_which=None):
	n_samples = 0
	n_acc_samples = 0
	n_nonacc_samples = 0
	samples_grouped_by_frame = []

	for s, scene in enumerate(scenes):
		for f, frame in enumerate(scene):

			samples_per_frame = []
			if f < frames_per_sample-1:
				continue

			frame_t_minus_0_info = json.load(open(scene[f]['json_path']))
			frame_t_minus_1_info = json.load(open(scene[f-1]['json_path']))
			frame_t_minus_2_info = json.load(open(scene[f-2]['json_path']))
			frame_t_minus_3_info = json.load(open(scene[f-3]['json_path']))
			frame_t_minus_4_info = json.load(open(scene[f-4]['json_path']))
			frame_t_minus_5_info = json.load(open(scene[f-5]['json_path']))
			frame_label = scene[f]['frame_label']

			for bbox in frame_t_minus_0_info['vehicleInfo']:
				data_driven_label, rule_based_label, rule_based_prob = get_vehicle_label(bbox, motion_model, ttc_threshold)
				if data_driven_label != None and rule_based_label != None:
					sample = {
						'hashcode': bbox['hashcode'],
						'frame_label': frame_label,
						'frame_t-0.0s': frame_per_vehicle(scene[f], bbox),
						'frame_t-0.1s': frame_per_vehicle(
											scene[f-1], 
											get_bbox(frame_t_minus_1_info,bbox['hashcode']))
										if frames_per_sample > 1 else None ,
						'frame_t-0.2s': frame_per_vehicle(
											scene[f-2], 
											get_bbox(frame_t_minus_2_info,bbox['hashcode']))
										if frames_per_sample > 2 else None ,
						'frame_t-0.3s': frame_per_vehicle(
											scene[f-3], 
											get_bbox(frame_t_minus_3_info,bbox['hashcode']))
										if frames_per_sample > 3 else None ,
						'frame_t-0.4s': frame_per_vehicle(
											scene[f-4], 
											get_bbox(frame_t_minus_4_info,bbox['hashcode']))
										if frames_per_sample > 4 else None,
						'frame_t-0.5s': frame_per_vehicle(
											scene[f-5], 
											get_bbox(frame_t_minus_5_info,bbox['hashcode']))
										if frames_per_sample > 5 else None,
					}
					n_samples += 1
					if label_method == "data_driven":
						sample['label'] = data_driven_label
						sample['label_prob'] = data_driven_label	
						if data_driven_label:
							n_acc_samples += 1 
						else:
							n_nonacc_samples += 1
					elif label_method == "rule_based":
						sample['label'] = rule_based_label	
						sample['label_prob'] = rule_based_label	
						if rule_based_label:
							n_acc_samples += 1  
						else: 
							n_nonacc_samples += 1
					elif label_method == "rule_based_prob":
						sample['label'] = rule_based_label	
						sample['label_prob'] = rule_based_prob	
						if rule_based_label:
							n_acc_samples += 1  
						else: 
							n_nonacc_samples += 1
					samples_per_frame.append(sample)

			samples_grouped_by_frame.append(samples_per_frame)

		util.over_print("Building sample list: %d/%d \r" %(s, len(scenes)))
	print()

	return samples_grouped_by_frame, n_acc_samples, n_nonacc_samples, n_samples


class Dataset():
	def initialize(self, opt, sess):
		self.opt = opt
		self.sess = sess
		self.train_root = opt.train_root
		self.valid_root = opt.valid_root
		self.test_root = opt.test_root
		self.n_test_splits = opt.n_test_splits
		self.train_frames_per_scene = opt.train_frames_per_scene
		self.valid_frames_per_scene = opt.valid_frames_per_scene
		self.test_frames_per_scene = opt.test_frames_per_scene
		self.train_frames_to_remove = opt.train_frames_to_remove
		self.valid_frames_to_remove = opt.valid_frames_to_remove
		self.test_frames_to_remove = opt.test_frames_to_remove
		self.frames_per_sample = opt.frames_per_sample
		self.no_shuffle_per_epoch = opt.no_shuffle_per_epoch
		self.label_method = opt.label_method
		self.motion_model = opt.motion_model
		self.ttc_threshold = opt.ttc_threshold
		self.pool = ThreadPool(16)  # Number of threads

		# load train dataset as a list of scenes 
		self.train_accident_scenes = build_scene_list(
										join(self.train_root,"accident/"), 
										self.train_frames_per_scene, 
										self.train_frames_to_remove, 
										data_label.ACCIDENT,
										)
		self.train_nonaccident_scenes = build_scene_list(
											join(self.train_root,"nonaccident/"), 
											self.train_frames_per_scene, 
											self.train_frames_to_remove, 
											data_label.NONACCIDENT,
											)
		self.train_scenes = self.train_accident_scenes + self.train_nonaccident_scenes
		
		# load validation dataset as a list of scenes 
		self.valid_accident_scenes = build_scene_list(
										join(self.valid_root,"accident/"), 
										self.valid_frames_per_scene, 
										self.valid_frames_to_remove, 
										data_label.ACCIDENT)
		self.valid_nonaccident_scenes = build_scene_list(
											join(self.valid_root,"nonaccident/"), 
											self.valid_frames_per_scene, 
											self.valid_frames_to_remove, 
											data_label.NONACCIDENT)
		self.valid_scenes = self.valid_accident_scenes + self.valid_nonaccident_scenes

		# shuffle training scenes without using the global random seed so that 
		# same dataset will be generated no matter which global random seed is given
		dataset_random = random.Random()
		dataset_random.seed(2018)
		dataset_random.shuffle(self.train_scenes)

		# use only part of the training dataset for training
		self.train_scenes = self.train_scenes[:int(self.opt.train_dataset_proportion*len(self.train_scenes))]

		# generate list of samples from list of scenes with corresponding options
		self.train_samples_grouped_by_frame, self.n_train_acc_samples, self.n_train_nonacc_samples, self.train_data_size = \
																			build_sample_list(
																					self.train_scenes, 
																					self.frames_per_sample, 
																					self.label_method, 
																					self.motion_model, 
																					self.ttc_threshold,
																					self.opt.shuffle_by_which)
		self.valid_samples_grouped_by_frame, self.n_valid_acc_samples, self.n_valid_nonacc_samples, self.valid_data_size = \
																			build_sample_list(
																					self.valid_scenes, 
																					self.frames_per_sample, 
																					self.label_method, 
																					self.motion_model, 
																					self.ttc_threshold)

		
		if self.opt.isTrain and self.n_train_acc_samples != 0:
			self.pos_ratio = self.n_train_nonacc_samples/self.n_train_acc_samples
		else:
			self.pos_ratio = 1

		self.train_steps_per_epoch = int(math.ceil(self.train_data_size/opt.batchSize))
		self.valid_steps_per_epoch = int(math.ceil(self.valid_data_size/opt.batchSize))

		self.test_datasets = []
		if self.n_test_splits > 0:
			test_roots = [join(self.test_root,str(split)) for split in range(self.n_test_splits)]
		else:
			test_roots = [self.test_root]
		for dataroot in test_roots:
			if dataroot != '':
				test_accident_scenes = build_scene_list(join(dataroot,"accident/"), 
														self.test_frames_per_scene, 
														self.test_frames_to_remove, 
														data_label.ACCIDENT)
				test_nonaccident_scenes = build_scene_list(join(dataroot,"nonaccident/"), 
															self.test_frames_per_scene, 
															self.test_frames_to_remove, 
															data_label.NONACCIDENT)
				test_scenes = test_accident_scenes + test_nonaccident_scenes
				test_samples_grouped_by_frame, n_test_acc_samples, n_test_nonacc_samples, test_data_size  = build_sample_list(
																			test_scenes, 
																			self.frames_per_sample, 
																			self.label_method, 
																			self.motion_model, 
																			self.ttc_threshold)
				self.test_datasets.append({
					"dataroot":dataroot,
					"n_test_acc_samples" : n_test_acc_samples,
					"n_test_nonacc_samples" : n_test_nonacc_samples,
					"test_samples_grouped_by_frame" : test_samples_grouped_by_frame,
					"test_data_size" : test_data_size,
					"test_steps_per_epoch" : int(math.ceil(test_data_size/opt.batchSize)),
					})

		print("# of train data: {} (acc: {} + non_acc: {})".format(
																self.train_data_size, 
																self.n_train_acc_samples, 
																self.n_train_nonacc_samples))
		print("# of valid data: {} (acc: {} + non_acc: {})".format(
																self.valid_data_size, 
																self.n_valid_acc_samples, 
																self.n_valid_nonacc_samples))
		for dataset in self.test_datasets:
			print("# of test data: {} (acc: {} + non_acc: {})".format(
																dataset['test_data_size'], 
																dataset['n_test_acc_samples'], 
																dataset['n_test_nonacc_samples']))

		# initialize queue operations for train dataset
		self.train_queue = tf.FIFOQueue(512, [tf.float32, tf.int32, tf.float32, tf.int32], shapes=[[130,355,opt.input_channel_dim], [], [], []])
		self.train_feature_input = tf.placeholder(tf.float32, shape=[None,130,355,opt.input_channel_dim])
		self.train_label_input = tf.placeholder(tf.int32, shape=[None])
		self.train_label_prob_input = tf.placeholder(tf.float32, shape=[None])
		self.train_frame_label_input = tf.placeholder(tf.int32, shape=[None])
		self.train_enqueue_op = self.train_queue.enqueue_many([self.train_feature_input,
																 self.train_label_input, 
																 self.train_label_prob_input,
																 self.train_frame_label_input])
		self.train_dequeue_size = tf.placeholder(tf.int32)
		self.train_feature_batch, self.train_label_batch, self.train_label_prob_batch, self.train_frame_label_batch = \
																		self.train_queue.dequeue_many(self.train_dequeue_size)

		# initialize queue operations for train dataset for evaluation
		self.train_eval_queue = tf.FIFOQueue(512, [tf.float32, tf.int32, tf.float32, tf.int32], shapes=[[130,355,opt.input_channel_dim], [], [], []])
		self.train_eval_feature_input = tf.placeholder(tf.float32, shape=[None,130,355,opt.input_channel_dim])
		self.train_eval_label_input = tf.placeholder(tf.int32, shape=[None])
		self.train_eval_label_prob_input = tf.placeholder(tf.float32, shape=[None])
		self.train_eval_frame_label_input = tf.placeholder(tf.int32, shape=[None])
		self.train_eval_enqueue_op = self.train_eval_queue.enqueue_many([self.train_eval_feature_input, 
																			self.train_eval_label_input,
																			self.train_eval_label_prob_input, 
																			self.train_eval_frame_label_input])
		self.train_eval_dequeue_size = tf.placeholder(tf.int32)
		self.train_eval_feature_batch, self.train_eval_label_batch, self.train_eval_label_prob_batch, self.train_eval_frame_label_batch = \
																		self.train_eval_queue.dequeue_many(self.train_eval_dequeue_size)

		# initialize queue operations for validation dataset
		self.valid_queue = tf.FIFOQueue(256, [tf.float32, tf.int32, tf.float32, tf.int32], shapes=[[130,355,opt.input_channel_dim], [], [], []])
		self.valid_feature_input = tf.placeholder(tf.float32, shape=[None,130,355,opt.input_channel_dim])
		self.valid_label_input = tf.placeholder(tf.int32, shape=[None])
		self.valid_label_prob_input = tf.placeholder(tf.float32, shape=[None])
		self.valid_frame_label_input = tf.placeholder(tf.int32, shape=[None])
		self.valid_enqueue_op = self.valid_queue.enqueue_many([self.valid_feature_input, 
																self.valid_label_input, 
																self.valid_label_prob_input, 
																self.valid_frame_label_input])
		self.valid_dequeue_size = tf.placeholder(tf.int32)
		self.valid_feature_batch, self.valid_label_batch, self.valid_label_prob_batch, self.valid_frame_label_batch =\
																	self.valid_queue.dequeue_many(self.valid_dequeue_size)

		# initialize queue operations for test dataset
		for d in self.test_datasets:
			d['test_queue'] = tf.FIFOQueue(512, [tf.float32, tf.int32, tf.float32, tf.int32], shapes=[[130,355,opt.input_channel_dim], [], [], []])
			d['test_feature_input'] = tf.placeholder(tf.float32, shape=[None,130,355,opt.input_channel_dim])
			d['test_label_input'] = tf.placeholder(tf.int32, shape=[None])
			d['test_label_prob_input'] = tf.placeholder(tf.float32, shape=[None])
			d['test_frame_label_input'] = tf.placeholder(tf.int32, shape=[None])
			d['test_enqueue_op'] = d['test_queue'].enqueue_many([d['test_feature_input'], 
																	d['test_label_input'], 
																	d['test_label_prob_input'], 
																	d['test_frame_label_input']])
			d['test_dequeue_size'] = tf.placeholder(tf.int32)
			d['test_feature_batch'], d['test_label_batch'], d['test_label_prob_batch'], d['test_frame_label_batch'] = \
																	d['test_queue'].dequeue_many(d['test_dequeue_size'])

		self.train_image_augmentator = None
		self.augment_bbox = False

	def preprocess_batch(self, batch, image_augmentator, augment_bbox):

		labels = [sample['label'] for sample in batch]
		label_probs = [sample['label_prob'] for sample in batch]
		frame_labels = [sample['frame_label'] for sample in batch]
		samples = self.pool.map(partial(self.preprocess_sample, image_augmentator=image_augmentator, augment_bbox=augment_bbox), batch)
		return samples, labels, label_probs, frame_labels

	def preprocess_sample(self, sample, image_augmentator, augment_bbox):

		sample_unstacked = []
		for i in range(self.opt.n_rgbs_per_sample):
			frame_t_minus_i = sample['frame_t-0.{}s'.format(i)]
			img_t_minus_i = self.load_and_process_img(frame_t_minus_i['img_path'], image_augmentator)
			sample_unstacked.append(img_t_minus_i)

		for i in range(self.opt.n_bbs_per_sample):
			frame_t_minus_i = sample['frame_t-0.{}s'.format(i)]
			bb_t_minus_i = self.generate_bbox_mask(frame_t_minus_i['bbox'], augment_bbox)
			sample_unstacked.append(bb_t_minus_i)

		x = np.concatenate(sample_unstacked,axis=2)

		return x

	def load_and_process_img(self, img_path, image_augmentator):
		if exists(splitext(img_path)[0]+".npy"):
			img = np.load(splitext(img_path)[0]+".npy")
		else:
			img = np.array(imread(img_path))
			if img.shape[0] == 400 and img.shape[1] == 710:
				img = img[100:360,:,0:3]
				img = imresize(img,(130,355))
			elif img.shape[0] == 130 and img.shape[1] == 355:
				pass
			else:
				ValueError("Undefined image size")
		if image_augmentator != None:
			img = image_augmentator.augment_image(img)
		return img

	def generate_bbox_mask(self, bbox, augment_bbox):
		mask = np.zeros((130,355,1))
		if bbox != None:
			x, y =int(bbox['x']), int(bbox['y'])
			width, height =int(bbox['width']), int(bbox['height']) 

			xmin, xmax = int(x/2), int((x+width)/2)
			ymin, ymax = int((y-100)/2), int((y-100+height)/2)

			if ymin < 0:
				ymin = 0
			if ymax < 0:
				ymax = 0
			if ymin > 129:
				ymin = 129
			if ymax > 129:
				ymax = 129
			if ymin == ymax:
				ymax=ymin+1
			if xmin == xmax:
				xmax=xmin+1
			mask[ymin:ymax,xmin:xmax,0] = 100
		return mask

	def init_train_generator(self):
		self.train_batch = self.train_batch_generator()

	def init_train_for_eval_generator(self):
		self.train_for_eval_batch = self.train_for_eval_batch_generator()

	def init_valid_generator(self):
		self.validation_batch = self.validation_batch_generator()

	def init_test_generator(self):
		for dataset in  self.test_datasets:
			dataset['test_batch'] = self.test_batch_generator(dataset)

	def train_batch_generator(self):
		for step in range(self.train_steps_per_epoch):
			train_data = self.dequeue_train_batch(self.opt.batchSize)
			yield train_data

	def train_for_eval_batch_generator(self):
		for step in range(self.train_steps_per_epoch):
			train_data = self.dequeue_train_for_eval_batch(self.opt.batchSize)
			yield train_data

	def validation_batch_generator(self):
		for step in range(self.valid_steps_per_epoch):
			if step+1 != self.valid_steps_per_epoch:
				dequeue_size = self.opt.batchSize
			else:
				dequeue_size = self.valid_data_size % self.opt.batchSize
			valid_data = self.dequeue_valid_batch(dequeue_size)
			yield valid_data

	def test_batch_generator(self, dataset):
		for step in range(dataset['test_steps_per_epoch']):
			if step+1 != dataset['test_steps_per_epoch']:
				dequeue_size = self.opt.batchSize
			else:
				dequeue_size = dataset['test_data_size'] % self.opt.batchSize
			test_data = self.dequeue_test_batch(dataset, dequeue_size)
			yield test_data

	def start_enqueue_threads_for_training(self, coord):
		self.enqueue_threads = []

		# Start threads to enqueue data asynchronously, and hide I/O latency.
		train_enqueue_thread = threading.Thread(
			target=self.sequential_load_and_enqueue,
			args=(coord, self.train_enqueue_op, self.train_samples_grouped_by_frame,
				self.train_feature_input, self.train_label_input, 
				self.train_label_prob_input, self.train_frame_label_input, 
				self.train_image_augmentator, self.augment_bbox
				),
			kwargs={'no_shuffle':self.no_shuffle_per_epoch}
			)
		self.enqueue_threads.append(train_enqueue_thread)

		train_eval_enqueue_thread = threading.Thread(
			target=self.sequential_load_and_enqueue,
			args=(coord, self.train_eval_enqueue_op, self.train_samples_grouped_by_frame,
				self.train_eval_feature_input, self.train_eval_label_input, 
				self.train_eval_label_prob_input, self.train_eval_frame_label_input, 
				),
			kwargs={'no_shuffle':True}
			)
		self.enqueue_threads.append(train_eval_enqueue_thread)

		valid_enqueue_thread = threading.Thread(
			target=self.sequential_load_and_enqueue,
			args=(coord, self.valid_enqueue_op, self.valid_samples_grouped_by_frame,
				self.valid_feature_input, self.valid_label_input, 
				self.valid_label_prob_input, self.valid_frame_label_input
				),
			kwargs={'no_shuffle':True}
			)
		self.enqueue_threads.append(valid_enqueue_thread)

		for dataset in self.test_datasets:
			test_enqueue_thread = threading.Thread(
				target=self.sequential_load_and_enqueue,
				args=(coord, dataset['test_enqueue_op'], dataset['test_samples_grouped_by_frame'],
					dataset['test_feature_input'], dataset['test_label_input'], 
					dataset['test_label_prob_input'], dataset['test_frame_label_input']
					),
				kwargs={'no_shuffle':True}
				)
			self.enqueue_threads.append(test_enqueue_thread)

		for t in self.enqueue_threads:
			t.start()

	def start_enqueue_threads_for_testing(self, coord):
		self.enqueue_threads = []

		for dataset in self.test_datasets:
			#Start threads to enqueue data asynchronously, and hide I/O latency.
			test_enqueue_thread = threading.Thread(target=self.sequential_load_and_enqueue,
				args=(coord, dataset['test_enqueue_op'], dataset['test_samples_grouped_by_frame'],
					dataset['test_feature_input'], dataset['test_label_input'], 
					dataset['test_label_prob_input'], dataset['test_frame_label_input']
					),
				kwargs={'no_shuffle':True}
				)
			self.enqueue_threads.append(test_enqueue_thread)

		for t in self.enqueue_threads:
			t.start()

	def sequential_load_and_enqueue(self, coord, enqueue_op, samples_grouped_by_frame, feature_placeholder, label_placeholder, label_prob_placeholder, frame_label_placeholder, image_augmentator=None, augment_bbox=False, no_shuffle=False):
		while not coord.should_stop():

			# shuffle samples in a frame-wise manner
			if not no_shuffle and self.opt.shuffle_by_which == "frame":
				shuffle(samples_grouped_by_frame)

			samples = list(itertools.chain.from_iterable(samples_grouped_by_frame))
			
			# shuffle samples in a sample-wise manner
			if not no_shuffle and self.opt.shuffle_by_which == "sample":
				shuffle(samples)

			sample_batches = list(chunks(samples, self.opt.batchSize))

			for input_batch in sample_batches:
			
				# Preprocess samples
				feature_input, label_input, label_prob_input, frame_label_input = self.preprocess_batch(input_batch, image_augmentator, augment_bbox)

				# Catch the exception that arises when you ask to close pending enqueue operations
				try:
					self.sess.run(enqueue_op, 
						feed_dict={
							feature_placeholder: feature_input, 
							label_placeholder: label_input, 
							label_prob_placeholder: label_prob_input, 
							frame_label_placeholder: frame_label_input
						})
				except tf.errors.CancelledError:
					return

	def dequeue_train_batch(self, dequeue_size):
		X_train, y_train,  yp_train, fy_train = self.sess.run(
				[self.train_feature_batch, self.train_label_batch, self.train_label_prob_batch, self.train_frame_label_batch], 
				feed_dict={self.train_dequeue_size: dequeue_size})
		return {'X':X_train, 'label':y_train, 'label_prob':yp_train, 'frame_label':fy_train}

	def dequeue_train_for_eval_batch(self, dequeue_size):
		X_train, y_train, yp_train, fy_train = self.sess.run(
				[self.train_eval_feature_batch, self.train_eval_label_batch, self.train_eval_label_prob_batch, self.train_eval_frame_label_batch], 
				feed_dict={self.train_eval_dequeue_size: dequeue_size})
		return {'X':X_train, 'label':y_train, 'label_prob':yp_train, 'frame_label':fy_train}

	def dequeue_valid_batch(self, dequeue_size):
		X_valid, y_valid,  yp_valid, fy_valid = self.sess.run(
				[self.valid_feature_batch, self.valid_label_batch, self.valid_label_prob_batch, self.valid_frame_label_batch], 
				feed_dict={self.valid_dequeue_size: dequeue_size})
		return {'X':X_valid, 'label':y_valid, 'label_prob':yp_valid, 'frame_label':fy_valid}

	def dequeue_test_batch(self, dataset, dequeue_size):
		X_test, y_test, yp_test, fy_test = self.sess.run(
				[dataset['test_feature_batch'], dataset['test_label_batch'], dataset['test_label_prob_batch'], dataset['test_frame_label_batch']], 
				feed_dict={dataset['test_dequeue_size']: dequeue_size})
		return {'X':X_test, 'label':y_test, 'label_prob':yp_test, 'frame_label':fy_test}

	def close_threads(self, coord):
		coord.request_stop()
		self.sess.run(self.train_queue.close(cancel_pending_enqueues=True))
		self.sess.run(self.valid_queue.close(cancel_pending_enqueues=True))
		for dataset in self.test_datasets:
			self.sess.run(dataset['test_queue'].close(cancel_pending_enqueues=True))
		coord.join(self.enqueue_threads, stop_grace_period_secs=10)

	def name(self):
		return 'SamplePerVehicleDataset'
