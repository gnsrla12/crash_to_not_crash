import os
os.system("export DISPLAY=:0.0")

#######################################################
####       Train with GTACrash dataset            #####
#######################################################

seed = 0
ttc = 1.8
feature_extractor = 'vgg16'
optimizer = 'adam'
lr = 0.0001
decay_lr_per = 3
nepoch = 10
init_steps_to_skip_eval = 1500

for train_proportion in [1.0]:

	for label_method in ['rule_based']:
	
		for motion_model in ['ctra']:

			for n_RGBs, n_BBs in [(3,3)]:

				os.system("python train.py " + \
							"--name temp.gtaCrash.{train_proportion}.t-{ttc}s/{label_method}-{motion_model}/".format(
											train_proportion=train_proportion, ttc=ttc, 
											label_method=label_method, motion_model=motion_model) +\
									"{n_RGBs}rgb{n_BBs}b.{feature_extractor}.{optimizer}.lr{lr}.decay_lr_per{decay_lr_per}.seed{seed} ".format(
											n_RGBs=n_RGBs, n_BBs=n_BBs, feature_extractor=feature_extractor,
											optimizer=optimizer, lr=lr, decay_lr_per=decay_lr_per, seed=seed)+ \
								"--train_root ./datasets/GTACrash/ "+ \
								"--valid_root ./datasets/YouTubeCrash/test/ "+ \
								"--test_root ./datasets/YouTubeCrash/test "+ \
								"--train_dataset_proportion {} ".format(train_proportion)+ \
								"--train_frames_per_scene 25 "+ \
								"--valid_frames_per_scene 20 "+ \
								"--test_frames_per_scene 20 "+ \
								"--label_method {} ".format(label_method)+ \
								"--motion_model {} ".format(motion_model)+ \
								"--ttc_threshold {} ".format(ttc)+ \
								"--n_rgbs_per_sample {} ".format(n_RGBs)+ \
								"--n_bbs_per_sample {} ".format(n_BBs)+ \
								"--feature_extractor {} ".format(feature_extractor)+ \
								"--optimizer {} ".format(optimizer)+ \
								"--lr {} ".format(lr)+ \
								"--decay_lr_per {} ".format(decay_lr_per)+ \
								"--nepoch {} ".format(nepoch)+ \
								"--batch_norm "+ \
								"--init_steps_to_skip_eval {} ".format(init_steps_to_skip_eval)+ \
								"--n_train_samples_to_eval 256 "+ \
								"--n_valid_samples_to_eval 0 "+ \
								"--seed {} ".format(seed)
						)

