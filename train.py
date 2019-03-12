import time, random
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

import utils.util as util
from dataloader.dataloader import CreateDataLoader
from models.models import CreateModel
from options.train_options import TrainOptions
from utils.simple_logger import SimpleLogger

opt = TrainOptions().parse()

gpu_config = tf.ConfigProto(device_count = {'GPU': opt.gpu_count})

np.random.seed(opt.seed)
random.seed(opt.seed)
tf.set_random_seed(opt.seed)

def evaluate_model(model, batches, steps_per_epoch):
    probs = []
    labels = []
    eval_start_time = time.time()
    for step, batch in enumerate(batches):
        if step >= steps_per_epoch:
            break
        labels.extend(batch['label'])
        probs.extend(model.forward(batch['X']))
        util.over_print("Step: %d/%d \tTime: %f\r" % (step+1, steps_per_epoch, time.time()-eval_start_time))
    loss, roc_auc, accuracy = util.measure_performance(labels, probs)
    return loss, roc_auc, accuracy

with tf.Session(config=gpu_config) as sess:
    
    dataloader = CreateDataLoader(opt, sess)
    coord = tf.train.Coordinator()
    dataloader.start_enqueue_threads_for_training(coord) 
    model = CreateModel(opt, sess, pos_weight=dataloader.pos_ratio)
    logger = SimpleLogger(opt)
    
    total_steps = -1
    best_valid_roc_auc = 0
    best_test_roc_auc = [0]*len(dataloader.test_datasets)

    if opt.load_epoch == "best_test":
        load_epoch = 0
    else:
        load_epoch = int(opt.load_epoch[-1])

    for epoch in range(0, opt.nepoch):

        # Skip epoch to continue training from specifed load epoch
        if opt.continue_train and epoch < load_epoch:
            continue

        # Optimize Model on Training Data
        if epoch > load_epoch:

            print('======================= Start of epoch %d ========================'%(epoch))
            epoch_start_time = time.time()
            step_start_time = time.time()
            dataloader.init_train_generator()

            for step, batch in enumerate(dataloader.train_batch):
                total_steps += 1
                current_epoch = total_steps/dataloader.train_steps_per_epoch
                
                # Update learning rate
                if opt.decay_lr_per > 0:
                    denominator = 10**((current_epoch)//opt.decay_lr_per)
                    lr = opt.lr/denominator
                else:
                    lr = opt.lr

                if (total_steps > opt.init_steps_to_skip_eval and total_steps%opt.eval_freq==0) or (step == dataloader.train_steps_per_epoch-1) or (total_steps==0):     
                    dataloader.init_train_for_eval_generator()
                    dataloader.init_valid_generator()
                    dataloader.init_test_generator()  

                    print("\n------------------- Current epoch: %f -------------------"%(current_epoch))

                    # Measure Performance on Partial Train Data
                    print("Measuring Partial Train Loss and Accuracy...")
                    train_loss, train_roc_auc, train_accuracy = evaluate_model(model, dataloader.train_for_eval_batch, int(opt.n_train_samples_to_eval/opt.batchSize))
                    print("\nTrain - Loss: %f \t  - ROC-AUC: %f \t Accuracy: %f \n"%(train_loss, train_roc_auc, train_accuracy))

                    if opt.n_valid_samples_to_eval > 0:
                        # Measure Performance on Validation Dataset
                        print("Measuring Validation Loss and Accuracy...")
                        valid_loss, valid_roc_auc, valid_accuracy = evaluate_model(model, dataloader.validation_batch, int(opt.n_valid_samples_to_eval/opt.batchSize))
                        print("\nValidation - Loss: %f\t ROC-AUC: %f\t Accuracy: %f\n"%(valid_loss, valid_roc_auc, valid_accuracy))

                        # Save current epoch as best epoch if it achieves best validation or test roc-auc
                        if best_valid_roc_auc < valid_roc_auc and current_epoch > 0:
                            best_valid_roc_auc = valid_roc_auc
                            model.save_model_with_name("best_validation", current_epoch, valid_roc_auc, test_roc_auc)
                    else:
                        valid_loss, valid_roc_auc, valid_accuracy = 0,0,0

                    # Measure Performance on Test Datasets
                    print("Measuring Test Loss and Accuracy...")
                    test_losses = []
                    test_roc_aucs = []
                    for i, dataset in enumerate(dataloader.test_datasets):
                        print("Data root: ",dataset['dataroot'])
                        test_loss, test_roc_auc, test_accuracy = evaluate_model(model, dataset['test_batch'], dataset['test_steps_per_epoch'])
                        print("\nTest - Loss: %f\t ROC-AUC: %f\t Accuracy: %f\n"%(test_loss, test_roc_auc, test_accuracy))
                        test_losses.append(test_loss)
                        test_roc_aucs.append(test_roc_auc)

                        if best_test_roc_auc[i] < test_roc_auc and current_epoch > 0:
                            best_test_roc_auc[i] = test_roc_auc
                            model.save_model_with_name("best_test_{}".format(i), current_epoch, valid_roc_auc, test_roc_auc)

                    # Log training results
                    logger.log({
                        "epoch":current_epoch,
                        "train_loss":train_loss,  "train_roc-auc":train_roc_auc,
                        "valid_loss":valid_loss, "valid_roc-auc":valid_roc_auc,
                        "test_loss":test_losses, "test_roc-auc":test_roc_aucs,
                        "best_test_roc-auc":list(best_test_roc_auc), "mean_best_test_roc-auc":np.mean(best_test_roc_auc),
                    })
                    print("\nBest Valid ROC-AUC: {}\t Best Test ROC-AUC: {} Mean of Best Test ROC-AUC: {}\n".format(
                                                        best_valid_roc_auc, best_test_roc_auc, np.mean(best_test_roc_auc)))
                
                # Take Gradient Descent
                loss = model.optimize_parameters(batch, lr)
                util.over_print("Train Step: %d/%d \t Loss: %f \t LR: %f \t Time: %f \r"
                    %(step, dataloader.train_steps_per_epoch, loss, lr, time.time() - step_start_time))
                step_start_time = time.time()


            print('\nEnd of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.nepoch, time.time()-epoch_start_time))
            # print('saving the model at the end of epoch %d'%(epoch))
            # model.save(epoch)

    dataloader.close_threads(coord)

    print("Done Training!")
    sess.close()
