#------------------------------------------------------------------------------
#	Import
#------------------------------------------------------------------------------
import torch, os
from glob import glob
import numpy as np
from collections import deque

from numpy import inf
from scipy.io import savemat
from time import time
from tqdm import tqdm


#------------------------------------------------------------------------------
#	Model-training template on an epoch
#------------------------------------------------------------------------------
def train_on_epoch(model, device, dataloader, loss_fn, optimizer, epoch):
	# Setup
	time_start = time()
	running_loss = 0
	running_corrects = 0
	n_steps = len(dataloader)
	n_samples = len(dataloader.dataset)

	# Train
	model.train()
	print("Train on epoch %d" % (epoch))
	for (X, y) in tqdm(dataloader, total=n_steps):
		# Forward pass
		X, y = X.to(device), y.to(device)
		logits = model(X)
		loss = loss_fn(logits, y)
		_, preds = torch.max(logits.data, 1)
		iter_loss = loss.item()
		running_loss += iter_loss
		iter_correct = torch.sum(preds==y).item()
		running_corrects += iter_correct

		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	loss_train = running_loss / n_steps
	acc_train = running_corrects / n_samples
	time_exe = time() - time_start
	return loss_train, acc_train, time_exe


#------------------------------------------------------------------------------
#	Model-validating template within an epoch
#------------------------------------------------------------------------------
def valid_on_epoch(model, device, dataloader, loss_fn, epoch):
	# Setup
	time_start = time()
	running_loss = 0
	running_corrects = 0
	n_steps = len(dataloader)
	n_samples = len(dataloader.dataset)

	# Validate
	model.eval()
	print("Validate on epoch %d" % (epoch))
	for (X, y) in tqdm(dataloader, total=n_steps):
		# Forward pass
		X, y = X.to(device), y.to(device)
		logits = model(X)
		loss = loss_fn(logits, y)
		_, preds = torch.max(logits.data, 1)
		running_loss += loss.item()
		running_corrects += torch.sum(preds==y).item()

	loss_valid = running_loss / n_steps
	acc_valid = running_corrects / n_samples
	time_exe = time() - time_start
	return loss_valid, acc_valid, time_exe


#------------------------------------------------------------------------------
#	Checkpoint
#------------------------------------------------------------------------------
class CheckPoint_old(object):
	def __init__(self, model, folder, last_best_loss=inf):
		super(CheckPoint, self).__init__()
		self.model = model
		self.folder = folder
		self.best_loss = last_best_loss


	def backup(self, loss, metrics):
		if loss < self.best_loss:
			model_file = os.path.join(self.folder, "model-loss%.6f.ckpt" % (loss))
			metrics_file = os.path.join(self.folder, "metrics.mat")
			torch.save(self.model.state_dict(), model_file)
			savemat(metrics_file, metrics)

			print("Loss improved from %f to %f" % (self.best_loss, loss))
			print("Model is saved in", model_file)
			print("Metrics are saved in", metrics_file)

			self.best_loss = loss
		else:
			print("Loss is not improved from %f to %f" % (self.best_loss, loss))


class CheckPoint(object):
	def __init__(self, model, optimizer, loss_fn, savedir, improved_delta=0.01, last_best_loss=inf):
		super(CheckPoint, self).__init__()
		self.model = model
		self.optimizer = optimizer
		self.loss_fn = loss_fn
		self.savedir = savedir
		self.improved_delta = improved_delta
		self.best_loss = last_best_loss
		if not os.path.exists(savedir):
			os.makedirs(savedir)


	def backup(self, loss_train, loss_valid, acc_train, acc_valid, metrics, epoch):
		if (self.best_loss-loss_valid)>=self.improved_delta:
			checkpoint = {
				"epoch": epoch,
				"model": self.model.state_dict(),
				"optimizer": self.optimizer.state_dict(),
				"loss_fn": self.loss_fn,

				"loss_train": loss_train, "loss_valid": loss_valid,
				"acc_train": acc_train, "acc_valid": acc_valid,
			}
			fname_checkpoint = "checkpoint-epoch%d-loss%f.ckpt" % (epoch, loss_valid)
			file_checkpoint = os.path.join(self.savedir, fname_checkpoint)
			torch.save(checkpoint, file_checkpoint)

			file_metrics = os.path.join(self.savedir, "metrics.mat")
			savemat(file_metrics, metrics)

			print("Loss improved from %f to %f" % (self.best_loss, loss_valid))
			print("Checkpoint saved in", file_checkpoint)
			print("Metrics saved in", file_metrics)
			self.best_loss = loss_valid

		else:
			file_metrics = os.path.join(self.savedir, "metrics.mat")
			savemat(file_metrics, metrics)
			print("Metrics saved in", file_metrics)
			print("Not improved enough from %f" % (self.best_loss))


	def reload(self, last_best_loss, improved_delta):
		self.last_best_loss = last_best_loss
		self.improved_delta = improved_delta


#------------------------------------------------------------------------------
#	Early Stopping
#------------------------------------------------------------------------------
class EarlyStopping(object):
	def __init__(self, not_improved_thres=1,
					improved_delta=0.01,
					last_best_loss=inf):
		super(EarlyStopping, self).__init__()
		self.not_improved_thres = not_improved_thres
		self.improved_delta = improved_delta
		self.not_improved = 0
		self.best_loss = last_best_loss


	def check(self, loss):
		if (self.best_loss-loss)>=self.improved_delta:
			self.not_improved = 0
			self.best_loss = loss
		else:
			self.not_improved += 1
			print("Not improved enough quantity: %d times" % (self.not_improved))
			if self.not_improved==self.not_improved_thres:
				print("Early stopping")
				return True

		return False


	def reload(self, last_best_loss, not_improved_thres, improved_delta):
		self.not_improved = 0
		self.last_best_loss = last_best_loss
		self.not_improved_thres = not_improved_thres
		self.improved_delta = improved_delta


#------------------------------------------------------------------------------
#	Flat curve detection
#------------------------------------------------------------------------------
class FlatCurve(object):
	def __init__(self, buffer_len=10, var_thres=0.001):
		super(FlatCurve, self).__init__()
		self.buffer_len = buffer_len
		self.var_thres = var_thres
		self.buffer = deque([])


	def check(self, loss):
		# Write data into the buffer
		if len(self.buffer)==self.buffer_len:
			self.buffer.pop()
			self.buffer.appendleft(loss)
		else:
			self.buffer.appendleft(loss)

		# Detect flat curve
		if len(self.buffer)==self.buffer_len:
			buff = list(self.buffer)
			max_val, min_val = max(buff), min(buff)
			if max_val-min_val<=self.var_thres:
				return True
				print("Detect loss is saturated")
			else:
				return False


	def reload(self, buffer_len, var_thres):
		self.buffer_len = buffer_len
		self.var_thres = var_thres
		self.buffer = deque([])


#------------------------------------------------------------------------------
#	Get the last best checkpoint
#------------------------------------------------------------------------------
def get_best_checkpoint(savedir):
	files = glob(os.path.join(savedir, "*.*"))
	files = [file for file in files if "checkpoint" in file]
	files = sorted(files)
	# print("Number of checkpoints:", len(files))

	best_loss = np.inf
	for idx, file in enumerate(files):
		fname = file.split("/")[-1]
		name = ".".join(fname.split(".")[:-1])
		fields = name.split("-")
		for field in fields:
			if "loss" in field:
				loss = float(field.replace("loss", ""))
				if loss <= best_loss:
					best_loss = loss
					best_idx = idx

	return files[best_idx]


#------------------------------------------------------------------------------
#	Load checkpoint
#------------------------------------------------------------------------------
def load_checkpoint(checkpoint_file, model, optimizer, loss_fn):
	checkpoint = torch.load(checkpoint_file)
	epoch = checkpoint["epoch"]
	model.load_state_dict(checkpoint["model"])
	optimizer.load_state_dict(checkpoint["optimizer"])
	loss_fn = checkpoint["loss_fn"]

	loss_train = checkpoint["loss_train"]
	loss_valid = checkpoint["loss_valid"]
	acc_train = checkpoint["acc_train"]
	acc_valid = checkpoint["acc_valid"]

	print("Load checkpoint from %s" % (checkpoint_file))
	print("Last epoch:", epoch)
	print("Last loss_train:", loss_train)
	print("Last loss_valid:", loss_valid)
	print("Last acc_train:", acc_train)
	print("Last acc_valid:", acc_valid)

	return epoch, model, optimizer, loss_fn, loss_valid


#------------------------------------------------------------------------------
#  Decrease learning rate
#------------------------------------------------------------------------------
def decrease_lr(optimizer, fraction):
	for idx, g in enumerate(optimizer.param_groups):
		lr_old = g["lr"]
		lr_new = lr_old * fraction
		g["lr"] = lr_new
		print("Change learning rate of group %d from %f to %f" % (idx, lr_old, lr_new))