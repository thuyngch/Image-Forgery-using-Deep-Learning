#------------------------------------------------------------------------------
#	Import
#------------------------------------------------------------------------------
import os
import torch
from multiprocessing import cpu_count


#------------------------------------------------------------------------------
#	Parameters
#------------------------------------------------------------------------------
params = {}

# Device
params["device"] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# CASIA-v2 directory struture
params["casia2_au"] = "CASIA-v2/Au/"
params["casia2_tp"] = "CASIA-v2/Tp/"


# Information of patches
params["label_dir"] = "data/"
params["neg_patches_file"] = os.path.join(params["label_dir"], "patch_coord_neg_casiav2.txt")
params["pos_patches_file"] = os.path.join(params["label_dir"], "patch_coord_pos_casiav2.txt")

params["au_subsets_file"] = os.path.join(params["label_dir"], "au_subsets.json")
params["tp_subsets_file"] = os.path.join(params["label_dir"], "tp_subsets.json")

params["test_subset"] = 5 	# Choose one value in this list [0, 1, 2, 3, 4, 5]


# Patch-collecting parameters
params["channel"] = "YCbCr" 	# RGB, YCbCr
params["create_new_neg"] = True
params["patch_sz"] = 64
params["stride"] = int(params["patch_sz"]/2)
params["n_workers"] = cpu_count()
params["pin_memory"] = True


# Storing images of patches
params["ratio"] = 0.90

params["patch_dir"] = "patches/%s-%d/%s-%d-%d" % (
	params["channel"], params["patch_sz"],
	params["channel"], params["patch_sz"], params["test_subset"]
)
params["patch_train_dir"] = os.path.join(params["patch_dir"], "train")
params["patch_train_au_dir"] = os.path.join(params["patch_train_dir"], "au")
params["patch_train_tp_dir"] = os.path.join(params["patch_train_dir"], "tp")

params["patch_valid_dir"] = os.path.join(params["patch_dir"], "valid")
params["patch_valid_au_dir"] = os.path.join(params["patch_valid_dir"], "au")
params["patch_valid_tp_dir"] = os.path.join(params["patch_valid_dir"], "tp")

params["patch_test_dir"] = os.path.join(params["patch_dir"], "test")
params["patch_test_au_dir"] = os.path.join(params["patch_test_dir"], "au")
params["patch_test_tp_dir"] = os.path.join(params["patch_test_dir"], "tp")


# Logging
params["logging_dir"] = "logging/"


# Pre-trained model
params["log_dir"] = "logs/"
params["log_pretrain_dir"] = os.path.join(params["log_dir"], "pretrain/")
params["model_pretrain_file"] = os.path.join(params["log_pretrain_dir"], "model-pretrain.ckpt")
params["metrics_pretrain_file"] = os.path.join(params["log_pretrain_dir"], "metrics-pretrain.mat")


# Fine-tuned model
params["log_finetune_dir"] = os.path.join(params["log_dir"], "finetune/")
params["log_finetune_train_dir"] = os.path.join(params["log_finetune_dir"], "train/")
params["log_finetune_valid_dir"] = os.path.join(params["log_finetune_dir"], "valid/")
params["model_finetune_file"] = os.path.join(params["log_finetune_dir"], "model-finetune.ckpt")
params["metrics_finetune_file"] = os.path.join(params["log_finetune_dir"], "metrics-finetune.mat")


# Reconstruction
params["recons_dir"] = "recons/"
params["recons_au_dir"] = os.path.join(params["recons_dir"], "au/")
params["recons_tp_dir"] = os.path.join(params["recons_dir"], "tp/")


# Fine-tuning parameters
params["num_epoch_finetune"] = 1000
params["batch_sz_finetune"] = 32

# # FOR MOBILENET-V2
# params["batch_sz_finetune"] = 16
# # The first 15 epochs (from 1 to 15)
# params["lr_finetune"] = 1e-3
# params["wd_finetune"] = 0
# # The next 2 epochs (from 16 to 17)
# params["lr_finetune"] = 1e-4
# params["wd_finetune"] = 1e-5

params["lr_finetune"] = 0.01
params["wd_finetune"] = 1e-4

params["momentum"] = 0.9
params["nesterov"] = True

params["not_improved_finetune_iter"] = 5
params["improved_finetune_delta"] = 3e-5
params["lr_decay"] = 5


# Post-processing
params["surround"] = 8
params["threshold"] = 0.645
params["test_result_file"] = "logs/test-result.txt"


#------------------------------------------------------------------------------
#	Function prints parameters
#------------------------------------------------------------------------------
def print_params(params, pos=30, logging=None):
	printf = logging.info if logging else print
	for key, val in params.items():
		line = "\t%s"%(key) + " "*(pos-len(key)) + ": {}".format(val)
		printf(line)
	printf("")


#------------------------------------------------------------------------------
#	Check directories
#------------------------------------------------------------------------------
def check_directories(list_dirs):
	for dir in list_dirs:
		if not os.path.exists(dir):
			print("makedirs", dir)
			os.makedirs(dir)