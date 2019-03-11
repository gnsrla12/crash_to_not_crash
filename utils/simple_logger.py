from os.path import isfile, join

class SimpleLogger():
    def __init__(self, opt):
        self.opt = opt
        self.save_dir = join(self.opt.checkpoints_dir, self.opt.name)

        # Measurements used for graphing loss and auc.
        self.batches = []
        
        self.train_loss_batch = []
        self.train_rocauc_batch = []
        
        self.valid_loss_batch = []
        self.valid_rocauc_batch = []

        self.test_loss_batch = []
        self.test_rocauc_batch = []

        self.best_test_rocauc_batch = []
        self.mean_best_test_rocauc_batch = []

                
    def log(self, log):
        # Log batches
        self.batches.append(log['epoch'])
        self.train_loss_batch.append(log['train_loss'])
        self.train_rocauc_batch.append(log['train_roc-auc'])
        self.valid_loss_batch.append(log['valid_loss'])
        self.valid_rocauc_batch.append(log['valid_roc-auc'])
        self.test_loss_batch.append(log['test_loss'])
        self.test_rocauc_batch.append(log['test_roc-auc'])
        self.best_test_rocauc_batch.append(log['best_test_roc-auc'])
        self.mean_best_test_rocauc_batch.append(log['mean_best_test_roc-auc'])

        file = open(join(self.save_dir,"log.txt"),"w")
        for x in range(0,len(self.batches)):
            file.write("Epoch: "+str("%.4f" % self.batches[x])+\
                        " Train - Loss: "+str("%.4f" % self.train_loss_batch[x])+\
                        " ROC-AUC: "+str("%.4f" % self.train_rocauc_batch[x])+\
                        " Valid - Loss: "+str("%.4f" % self.valid_loss_batch[x])+\
                        " ROC-AUC: "+str("%.4f" % self.valid_rocauc_batch[x])+\
                        " Test - Losses: "+str(["%.4f"%loss for loss in self.test_loss_batch[x]])+\
                        " ROC-AUCs: "+str(["%.4f"%loss for loss in self.test_rocauc_batch[x]])+\
                        " Best ROC-AUCs: "+str(["%.4f"%loss for loss in self.best_test_rocauc_batch[x]])+\
                        " Mean Best ROC-AUC: "+str("%.4f" % self.mean_best_test_rocauc_batch[x])+\
                        "\n")
        file.close() 

    def name(self):
        return 'SimpleLogger'
