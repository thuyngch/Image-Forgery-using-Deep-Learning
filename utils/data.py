#------------------------------------------------------------------------------
#	Import
#------------------------------------------------------------------------------
import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import ImageFolder


#------------------------------------------------------------------------------
#	DataLoader: from image folder
#------------------------------------------------------------------------------
class ImageFolderLoader(object):
	def __init__(self,
				dir_image,
				color_channel="RGB",
				batch_size=32,
				n_workers=1,
				pin_memory=True,
				shuffle=True):

		# Storage parameters
		super(ImageFolderLoader, self).__init__()
		self.dir_image = dir_image
		self.color_channel = color_channel
		self.batch_size = batch_size
		self.n_workers = n_workers
		self.pin_memory = pin_memory
		self.shuffle = shuffle

	@property
	def transform(self):
		fnc_CvtChannel = lambda img_PIL: img_PIL.convert(self.color_channel)
		return torchvision.transforms.Compose([
			torchvision.transforms.Lambda(fnc_CvtChannel),
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225],
			)
		])

	@property
	def dataset(self):
		return ImageFolder(self.dir_image, transform=self.transform)

	@property
	def loader(self):
		return DataLoader(self.dataset,
						batch_size=self.batch_size,
						num_workers=self.n_workers,
						pin_memory=self.pin_memory,
						shuffle=self.shuffle)


#------------------------------------------------------------------------------
#	DataLoader: from Numpy image
#------------------------------------------------------------------------------
class NumpyImageLoader(object):
	"""
	ndarray_data : (ndarray) Numpy array of images of shape (3, W, H).
	"""
	def __init__(self,
				ndarray_data,
				ndarray_labels=None,
				batch_size=32,
				n_workers=1,
				pin_memory=True,
				shuffle=True):

		# Storage parameters
		super(NumpyImageLoader, self).__init__()
		self.labels = ndarray_labels
		self.batch_size = batch_size
		self.n_workers = n_workers
		self.pin_memory = pin_memory
		self.shuffle = shuffle

		# Preprocess data
		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]
		self.data = torch.tensor(ndarray_data/255, dtype=torch.float32)
		for c in range(3):
			self.data[c,...] = (self.data[c,...] - mean[c]) / std[c]

	@property
	def dataset(self):
		if self.labels:
			return TensorDataset(self.data, self.labels)
		else:
			return TensorDataset(self.data)

	@property
	def loader(self):
		return DataLoader(self.dataset,
						batch_size=self.batch_size,
						num_workers=self.n_workers,
						pin_memory=self.pin_memory,
						shuffle=self.shuffle)