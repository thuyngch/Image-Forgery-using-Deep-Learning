#------------------------------------------------------------------------------
#	Import
#------------------------------------------------------------------------------
import torch, datetime, argparse, os
from torch import nn
from torchsummary import summary
from multiprocessing import cpu_count
import numpy as np

from utils.MobileNetV2_pretrained_imagenet import MobileNetV2, ImageFolderLoader
from utils.logger import TensorboardLogger
from utils.learning import train_on_epoch, valid_on_epoch, CheckPoint, EarlyStopping


#------------------------------------------------------------------------------
#	Check directories
#------------------------------------------------------------------------------
def check_directories(list_dirs):
	for dir in list_dirs:
		if not os.path.exists(dir):
			print("makedirs", dir)
			os.makedirs(dir)


#------------------------------------------------------------------------------
#	Parse arguments
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
	'--channel', type=str, default="RGB",
	help='Color channel'
)
parser.add_argument(
	'--patch_sz', type=int, default=64,
	help='Patch size'
)
parser.add_argument(
	'--test_subset', type=int, default=5,
	help='Index of the test subset'
)
parser.add_argument(
	'--dir', type=str, default="./data/",
	help='Folder containing the extracted data'
)
parser.add_argument(
	'--n_epochs', type=int, default=50,
	help='Number of epochs'
)
args = parser.parse_args()


#------------------------------------------------------------------------------
#   Parameters
#------------------------------------------------------------------------------
# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training
BATCH_SZ = 64
LR = 1e-3
WD = 1e-5
MOMENTUM = 0.9

# Data directories
DIR = os.path.join(args.dir, "%s_%d_%d" % (
	args.channel, args.patch_sz, args.test_subset
))
TRAIN_DIR = os.path.join(DIR, "train")
VALID_DIR = os.path.join(DIR, "valid")

# TensorBoardX
TFBOARD_TRAIN_DIR = "models/MBN2-mod-%s/checkpoints/train" % (args.channel)
TFBOARD_VALID_DIR = "models/MBN2-mod-%s/checkpoints/valid" % (args.channel)

# Checkpoint
CHECKPOINT_DIR = "models/MBN2-mod-%s/checkpoints/" % (args.channel)
N_NOT_IMPROVED = 5
IMPROVED_DELTA = 1e-3

# Create directories
list_dirs = [
	CHECKPOINT_DIR,
	TFBOARD_TRAIN_DIR,
	TFBOARD_VALID_DIR,
]
check_directories(list_dirs)


#------------------------------------------------------------------------------
#	Setup
#------------------------------------------------------------------------------
# Data loader
train_loader = ImageFolderLoader(
	dir_image=TRAIN_DIR,
	color_channel=args.channel,
	batch_size=BATCH_SZ,
	n_workers=cpu_count(),
	pin_memory=True,
	shuffle=True,
).train_loader

valid_loader = ImageFolderLoader(
	dir_image=VALID_DIR,
	color_channel=args.channel,
	batch_size=BATCH_SZ,
	n_workers=cpu_count(),
	pin_memory=True,
	shuffle=False,
).valid_loader


# Create and load model
model = MobileNetV2(n_class=2, input_size=args.patch_sz, width_mult=1.0).to(DEVICE)
summary(model, input_size=(3, args.patch_sz, args.patch_sz))


# Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
	model.parameters(),
	lr=LR, weight_decay=WD,
	momentum=MOMENTUM, nesterov=True,
)


# Create logger
logger_train = TensorboardLogger(TFBOARD_TRAIN_DIR)
logger_valid = TensorboardLogger(TFBOARD_VALID_DIR)

metrics = {
	"loss_train": [],
	"loss_valid": [],
	"acc_train": [],
	"acc_valid": [],
}


# Create callbacks
checkpoint = CheckPoint(
	model=model,
	optimizer=optimizer,
	loss_fn=loss_fn,
	savedir=CHECKPOINT_DIR,
	improved_delta=IMPROVED_DELTA,
	last_best_loss=np.inf,
)
earlystop = EarlyStopping(
	not_improved_thres=N_NOT_IMPROVED,
	improved_delta=IMPROVED_DELTA,
)


#------------------------------------------------------------------------------
#   Train the model
#------------------------------------------------------------------------------
for epoch in range(1, args.n_epochs+1):
	print("------------------------------------------------------------------")
	# Train model
	loss_train, acc_train, time_train = train_on_epoch(
										model=model,
										device=DEVICE,
										dataloader=train_loader,
										loss_fn=loss_fn,
										optimizer=optimizer,
										epoch=epoch)

	logger_train.write_all_to_disk(epoch, loss_train, acc_train)
	print('loss_train: {}, acc_train: {}'.format(loss_train, acc_train))

	# Validate model
	loss_valid, acc_valid, time_valid = valid_on_epoch(
										model=model,
										device=DEVICE,
										dataloader=valid_loader,
										loss_fn=loss_fn,
										epoch=epoch)

	logger_valid.write_all_to_disk(epoch, loss_valid, acc_valid)
	print('loss_valid: {}, acc_valid: {}'.format(loss_valid, acc_valid))

	# Record and print
	metrics["loss_train"].append(loss_train)
	metrics["acc_train"].append(acc_train)
	metrics["loss_valid"].append(loss_valid)
	metrics["acc_valid"].append(acc_valid)
	print("Finish at {}, Runtime: {:.3f}[s]".format(datetime.datetime.now(), time_train+time_valid))

	# Callbacks
	checkpoint.backup(loss_train, loss_valid, acc_train, acc_valid, metrics, epoch)
	if earlystop.check(loss_valid):
		break