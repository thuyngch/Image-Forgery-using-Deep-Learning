#------------------------------------------------------------------------------
#    Libraries
#------------------------------------------------------------------------------
import os
import logging
from glob import glob

import tensorflow as tf
from scipy.io import savemat, loadmat
from matplotlib import pyplot as plt
from datetime import datetime


#------------------------------------------------------------------------------
#	Printed message logger
#------------------------------------------------------------------------------
class PrintLogger(object):
	"""
	log_dir : (str) Directory contains log files.
	filename : (str) Name of the executing file.
	max_num_file : (int) Maximum number of log files.
	"""
	def __init__(self, log_dir, filename, max_num_file=30):
		# Storage parameters
		super(PrintLogger, self).__init__()
		self.log_dir = log_dir
		self.filename = filename
		self.max_num_file = max_num_file

		# Check to create the log directory
		if not os.path.exists(log_dir):
			os.mkdir(self.log_dir)
			print("[PrintLogger] Create log directory at %s"%(self.log_dir))

		# Get information of existed log files in the directory
		self.log_files = glob(os.path.join(self.log_dir, "*.*"))
		self.n_log_files = len(self.log_files)

		# Create log file for the current running
		if (self.n_log_files>=self.max_num_file):
			self.remove_the_oldest_log_file()
		self.create_new_log_file()

		# Create a logging instance
		logging.basicConfig(
			level=logging.INFO,
			format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
			datefmt='%Y-%m-%d %H:%M',
			filename=self.cur_file,
			filemode='w',
		)
		console = logging.StreamHandler()
		console.setLevel(logging.INFO)
		logging.getLogger('').addHandler(console)
		self.logger = logging.getLogger(self.filename)

	def remove_the_oldest_log_file(self):
		fnames = [file.split("/")[-1] for file in self.log_files]
		names = [file.split(".")[0] for file in fnames]
		dtimes = [datetime.strptime(name, "%Y-%m-%d_%H:%M:%S") for name in names]
		times = [dtime.time() for dtime in dtimes]
		min_idx = times.index(min(times))
		old_file = self.log_files[min_idx]
		os.remove(old_file)
		self.log_files.remove(old_file)

	def create_new_log_file(self):
		cur_datetime = "_".join(str(datetime.now()).split())
		cur_datetime = cur_datetime.split(".")[0]
		self.cur_file = os.path.join(self.log_dir, "%s.log" % (cur_datetime))

	def info(self, string):
		self.logger.info(string)

	def warning(self, string):
		self.logger.warning(string)

	def error(self, string):
		self.logger.error(string)


#------------------------------------------------------------------------------
#    Tensorboard logger
#------------------------------------------------------------------------------
class TensorboardLogger(object):
	def __init__(self, log_dir):
		"""Create a summary writer logging to log_dir."""
		self.writer = tf.summary.FileWriter(log_dir)

	def scalar_summary(self, tag, value, step):
		"""Log a scalar variable"""
		summary = tf.Summary(
			value=[tf.Summary.Value(
				tag=tag,
				simple_value=value
			)]
		)
		self.writer.add_summary(summary, step)

	def write_all_to_disk(self, step, loss, acc):
		info = {'loss': loss, 'accuracy': acc}
		for tag, value in info.items():
			self.scalar_summary(tag, value, step+1)

	def write_loss_to_disk(self, step, loss):
		info = {'loss': loss}
		for tag, value in info.items():
			self.scalar_summary(tag, value, step+1)


#------------------------------------------------------------------------------
#    Metric logger
#------------------------------------------------------------------------------
class MetricLogger(object):
	def __init__(self, file):
		super(MetricLogger, self).__init__()
		self.file = file
		self.metrics = {}
		self.metrics["loss_train"] = []
		self.metrics["loss_valid"] = []
		self.metrics["acc_train"] = []
		self.metrics["acc_valid"] = []

	def update(self, loss_train, loss_valid, acc_train, acc_valid):
		self.metrics["loss_train"].append(loss_train)
		self.metrics["loss_valid"].append(loss_valid)
		self.metrics["acc_train"].append(acc_train)
		self.metrics["acc_valid"].append(acc_valid)

	def write(self):
		savemat(self.file, self.metrics)

	def load(self):
		metrics = loadmat(self.file)
		self.metrics["loss_train"] = metrics["loss_train"][0].tolist()
		self.metrics["loss_valid"] = metrics["loss_valid"][0].tolist()
		self.metrics["acc_train"] = metrics["acc_train"][0].tolist()
		self.metrics["acc_valid"] = metrics["acc_valid"][0].tolist()

	def visualize(self):
		plt.figure(1)

		plt.subplot(1,2,1); plt.axis("on"); plt.title("Loss")
		plt.plot(self.metrics["loss_train"], "-*r")
		plt.plot(self.metrics["loss_valid"], "-+b")
		plt.legend(["train", "valid"])

		plt.subplot(1,2,2); plt.axis("on"); plt.title("Accuracy")
		plt.plot(self.metrics["acc_train"], "-*r")
		plt.plot(self.metrics["acc_valid"], "-+b")
		plt.legend(["train", "valid"])

		plt.show()