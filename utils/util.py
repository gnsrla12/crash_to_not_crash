from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from PIL import Image
import numpy as np
import data_label
import os
import sys

import matplotlib
from matplotlib import pyplot as plt

def measure_performance(labels, probs):
    preds = [np.argmax(p) for p in probs]
    acc_probs = [p[1] for p in probs]
    try:
        loss = log_loss(labels, probs)
    except:
        print("Error Calculating loss")
        loss = 0
    try:
        roc_auc = roc_auc_score(labels, acc_probs)
    except:
        print("Error Calculating roc-auc")
        roc_auc = 0
    try:
        accuracy = accuracy_score(labels, preds)
    except:
        print("Error Calculating accuracy")
        accuracy = 0
    return loss, roc_auc, accuracy
    
def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def show_image(data):
    plt.imshow(data)
    plt.show()

def show_sample(x,label):
    plt.imshow(x)
    plt.xlabel(data_label.class_type[label])
    plt.show()

def over_print(text):
    sys.stdout.write(text)
    sys.stdout.flush()

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
