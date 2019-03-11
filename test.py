import os, time, random, itertools
import tensorflow as tf
import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt

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

np.random.seed(opt.seed)
random.seed(opt.seed)
tf.set_random_seed(opt.seed)

result_dict = {
    "split_0_result": None,
    "split_1_result": None,
    "split_2_result": None,
    "split_3_result": None,
    "split_4_result": None,
}

with tf.Session(config=gpu_config) as sess:

    model = CreateModel(opt, sess)   

    data_loader = CreateDataLoader(opt, sess)
    coord = tf.train.Coordinator()
    data_loader.start_enqueue_threads_for_testing(coord)
    data_loader.init_test_generator()

    # Measure Performance on Test Datasets
    print("Measuring Test Loss and Accuracy...")
    test_roc_aucs = []
    for split, dataset in enumerate(data_loader.test_datasets):

        if not opt.load_epoch:
            model.load("best_test_{}".format(split))

        print("Data root: ",dataset['dataroot'])
        
        probs = []
        labels = []
        eval_start_time = time.time()
        for step, batch in enumerate(dataset['test_batch']):
            labels.extend(batch['label'])
            probs.extend(model.forward(batch['X']))

            if opt.plot:
                print("Accident Probability: %f"%probs[-1][1])           
                print("Label: %f"%labels[-1])           
                print("Frame Label: %f"%batch['frame_label'][-1])           
                frame = batch['X'][-1]     
                image = frame[:,:,0:3].astype(np.uint8)
                bbox2 = frame[:,:,6].astype(np.uint8)
                masked2 = np.ma.masked_where(bbox2 == 0, bbox2)
                bbox1 = frame[:,:,7].astype(np.uint8)
                masked1 = np.ma.masked_where(bbox1 == 0, bbox1)
                plt.imshow(image)
                plt.imshow(masked2, 'Reds_r', alpha=0.4)
                plt.imshow(masked1, 'jet', alpha=0.4)
                plt.show()

            util.over_print("Step: %d/%d \tTime: %f\r" % (step+1, dataset['test_steps_per_epoch'], time.time()-eval_start_time))


        samples_grouped_by_frame = dataset['test_samples_grouped_by_frame']
        samples = list(itertools.chain.from_iterable(samples_grouped_by_frame))
        result_dict['split_{}_result'.format(split)] = {
                                                        'labels' : labels, 
                                                        'probs' : probs,
                                                        'samples' : samples
                                                        }


        test_loss, test_roc_auc, test_accuracy = util.measure_performance(labels, probs)
        test_roc_aucs.append(test_roc_auc)        
        print("\nTest - Loss: %f\t ROC-AUC: %f\t Accuracy: %f\n"%(test_loss, test_roc_auc, test_accuracy))
    
    print("Test ROC-AUC: ", test_roc_aucs)
    print("Mean Test ROC-AUC: ", np.mean(test_roc_aucs))
    print()

    data_loader.close_threads(coord)
    sess.close()



