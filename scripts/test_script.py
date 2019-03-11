import os
os.system("export DISPLAY=:0.0")

#######################################################
####       Test on YouTubeCrash dataset            #####
#######################################################

seed = 0
ttc = 1.8
feature_extractor = "vgg16"
optimizer = 'adam'
lr = 0.0001
decay_lr_per = 1

for train_proportion in [1.0]:

	for train_label_method in ["rule_based"]:

		for motion_model in ['ctra']:

			for n_RGBs, n_BBs in [(3,3)]:

				os.system("python test.py "+ \
							"--name gtaCrash.{train_proportion}.t-{ttc}s/{label_method}-{motion_model}/".format(
											train_proportion=train_proportion, ttc=ttc, 
											label_method=train_label_method, motion_model=motion_model) +\
									"{n_RGBs}rgb{n_BBs}b.{feature_extractor}.{optimizer}.lr{lr}.decay_lr_per{decay_lr_per}.seed{seed} ".format(
											n_RGBs=n_RGBs, n_BBs=n_BBs, feature_extractor=feature_extractor, optimizer=optimizer,
											lr=lr, decay_lr_per=decay_lr_per, seed=seed)+ \
							"--test_root ./datasets/YouTubeCrash/test "+ \
							"--load_epoch best_test "+ \
							"--label_method data_driven "+ \
							"--test_frames_per_scene 20 "+ \
							"--ttc_threshold {} ".format(ttc)+ \
							"--feature_extractor {} ".format(feature_extractor)+ \
							"--n_rgbs_per_sample {} ".format(n_RGBs)+ \
							"--n_bbs_per_sample {} ".format(n_BBs)+ \
							"--batch_norm "
					)




