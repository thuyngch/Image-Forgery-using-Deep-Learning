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
params["patch_dir"] = "patches/%s-%d/%s-%d-%d" % (
	params["channel"], params["patch_sz"],
	params["channel"], params["patch_sz"], params["test_subset"]
)
params["ratio"] = 0.90

params["patch_train_dir"] = os.path.join(params["patch_dir"], "train")
params["patch_train_au_dir"] = os.path.join(params["patch_train_dir"], "au")
params["patch_train_tp_dir"] = os.path.join(params["patch_train_dir"], "tp")

params["patch_valid_dir"] = os.path.join(params["patch_dir"], "valid")
params["patch_valid_au_dir"] = os.path.join(params["patch_valid_dir"], "au")
params["patch_valid_tp_dir"] = os.path.join(params["patch_valid_dir"], "tp")


# Logging
params["logging_dir"] = "logging/"


# Model log
params["model_dir"] = "models/mobilenetv2_pretrained_imagenet/"
params["suffix"] = "%s-%d-%d" % (params["channel"], params["patch_sz"], params["test_subset"])

params["training_log_dir"] = os.path.join(params["model_dir"], "logs-%s" % (params["suffix"]))

params["recons_dir"] = os.path.join(params["model_dir"], "recons-%s" % (params["suffix"]))
params["recons_au_dir"] = os.path.join(params["recons_dir"], "au")
params["recons_tp_dir"] = os.path.join(params["recons_dir"], "tp")

params["test_ft_dir"] = os.path.join(params["model_dir"], "test-%s" % (params["suffix"]))
params["test_ft_au_dir"] = os.path.join(params["test_ft_dir"], "au")
params["test_ft_tp_dir"] = os.path.join(params["test_ft_dir"], "tp")

params["test_result_file"] = os.path.join(params["model_dir"], "test-result-%s.txt" % (params["suffix"]))


# Training configurations
params["num_epoch"] = 1000
params["batch_sz"] = 32

params["lr"] = 0.001
params["wd"] = 2e-4
params["momentum"] = 0.9
params["nesterov"] = True

params["not_improved_iter"] = 5
params["improved_delta"] = 0.00001


# Post-processing
params["surround"] = 8
params["threshold"] = 0.8