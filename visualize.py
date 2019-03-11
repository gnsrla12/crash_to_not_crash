import os, time, json, cv2
from os.path import basename, join
import tensorflow as tf
import numpy as np
from sklearn import metrics
from scipy.misc import imread, imsave
from matplotlib import pyplot as plt
import itertools

import utils.util as util
from options.test_options import TestOptions
from models.models import CreateModel
from dataloader.dataloader import CreateDataLoader
import data_label

opt = TestOptions().parse()
opt.batchSize = 128  

gpu_config = tf.ConfigProto(
        device_count = {'GPU': opt.gpu_count}
    )

dst = "./visualization/"
util.mkdirs(dst)

################################################
#   Acquire prediction results on all samples  #
################################################
with tf.Session(config=gpu_config) as sess:

    model = CreateModel(opt, sess)   
    data_loader = CreateDataLoader(opt, sess)
    coord = tf.train.Coordinator()
    data_loader.start_enqueue_threads_for_testing(coord)
    data_loader.init_test_generator()
    y_true = []
    y_pred = []
    step_count = 0.0

    for step, batch in enumerate(data_loader.test_datasets[0]['test_batch']):
        probs = model.forward(batch['X'])
        y_true.extend(batch['label'])
        y_pred.extend(probs)
        step_count += 1
        util.over_print("Step count: {} \r".format(str(step_count)))

    temp_y_pred = [p[1] for p in y_pred]
    data_loader.close_threads(coord)
    sess.close()

####################################################################################
#   Filter only top n vehicles within a frame with highest collision probability   #
####################################################################################
if opt.filter_top_n > 0:
    samples_grouped_by_frame = data_loader.test_datasets[0]['test_samples_grouped_by_frame']
    samples = list(itertools.chain.from_iterable(samples_grouped_by_frame))
    json_path = samples[0]['frame_t-0.0s']['json_path']
    new_y_pred = []
    preds_per_frame = []

    for j, (sample, pred) in enumerate(zip(samples, y_pred)):
        if json_path != sample['frame_t-0.0s']['json_path'] or j==len(samples)-1:
            if j == len(samples)-1:
                preds_per_frame.append(pred)

            json_path = sample['frame_t-0.0s']['json_path']
            max_index = np.argsort(-np.asarray(preds_per_frame)[:,1])[0:opt.filter_top_n]
            filtered_probs = np.zeros_like(preds_per_frame)
            preds_per_frame = np.asarray(preds_per_frame)
            filtered_probs[max_index] = preds_per_frame[max_index]
            new_y_pred.extend(filtered_probs)
            preds_per_frame = []

        preds_per_frame.append(pred)        
    y_pred = new_y_pred

#############################################
#   Visualize the collision probabilities   #
#############################################
samples_grouped_by_frame = data_loader.test_datasets[0]['test_samples_grouped_by_frame']
samples = list(itertools.chain.from_iterable(samples_grouped_by_frame))
json_path = samples[0]['frame_t-0.0s']['json_path']
json_path = ""
img = ""
img_count = 0

for i, (sample, pred) in enumerate(zip(samples, y_pred)):
    p = pred[1]
    util.over_print("sample: {}/{} \r".format(i,len(samples)))

    if json_path != sample['frame_t-0.0s']['json_path']:
        if img != "":
            imsave(join(dst,"{}.jpg".format(str(img_count))),img)
            img_count += 1
        json_path = sample['frame_t-0.0s']['json_path']
        img_path = sample['frame_t-0.0s']['img_path']

        # Load Image
        img = imread(img_path)
        # Read Frame Information from Json File
        with open(json_path, 'r') as file:
            FrameInfo = json.load(file)

    vehicle_dict = sample['frame_t-0.0s']['bbox']
    x,y,width,height = int(vehicle_dict['x']), int(vehicle_dict['y']), int(vehicle_dict['width']), int(vehicle_dict['height'])
    collidingObj = vehicle_dict['label']

    # Mark Ground Truth Colliding Vehicle in BLUE
    if collidingObj == True:
        line_width = 5
        color = [0,0,255]
        cv2.line(img, (x, y), (x + width, y), color, line_width)
        cv2.line(img, (x + width, y), (x + width, y+height), color, line_width)
        cv2.line(img, (x, y+height), (x + width, y+height), color, line_width)
        cv2.line(img, (x, y), (x, y+height), color, line_width)    

    # Color BBox according to the predicted accident probability
    line_width = 2
    color = [255*(p),255*(1-p),0]
    cv2.line(img, (x, y), (x + width, y), color, line_width)
    cv2.line(img, (x + width, y), (x + width, y+height), color, line_width)
    cv2.line(img, (x, y+height), (x + width, y+height), color, line_width)
    cv2.line(img, (x, y), (x, y+height), color, line_width)
    cv2.putText(img,str("%.2f" % p), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
