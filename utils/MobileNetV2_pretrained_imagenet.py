import torch, math
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder



#------------------------------------------------------------------------------
#	Useful convolutional modules
#------------------------------------------------------------------------------
def conv_bn(inp, oup, stride):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
		nn.BatchNorm2d(oup),
		nn.ReLU6(inplace=True)
	)

def conv_1x1_bn(inp, oup):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
		nn.BatchNorm2d(oup),
		nn.ReLU6(inplace=True)
	)


#------------------------------------------------------------------------------
#	Inverted Residual block
#------------------------------------------------------------------------------
class InvertedResidual(nn.Module):
	def __init__(self, inp, oup, stride, expand_ratio):
		super(InvertedResidual, self).__init__()
		self.stride = stride
		assert stride in [1, 2]

		hidden_dim = round(inp * expand_ratio)
		self.use_res_connect = self.stride == 1 and inp == oup

		if expand_ratio == 1:
			self.conv = nn.Sequential(
				# dw
				nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
				nn.BatchNorm2d(hidden_dim),
				nn.ReLU6(inplace=True),
				# pw-linear
				nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup),
			)
		else:
			self.conv = nn.Sequential(
				# pw
				nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
				nn.BatchNorm2d(hidden_dim),
				nn.ReLU6(inplace=True),
				# dw
				nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
				nn.BatchNorm2d(hidden_dim),
				nn.ReLU6(inplace=True),
				# pw-linear
				nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup),
			)

	def forward(self, x):
		if self.use_res_connect:
			return x + self.conv(x)
		else:
			return self.conv(x)


#------------------------------------------------------------------------------
#	MobileNetV2
#------------------------------------------------------------------------------
class MobileNetV2(nn.Module):
	def __init__(self, n_class=1000, input_size=224, width_mult=1.):
		super(MobileNetV2, self).__init__()
		block = InvertedResidual
		input_channel = 32
		last_channel = 1280
		interverted_residual_setting = [
			# t, c, n, s
			[1, 16, 1, 1],
			[6, 24, 2, 2],
			[6, 32, 3, 2],
			[6, 64, 4, 2],
			[6, 96, 3, 1],
			[6, 160, 3, 2],
			[6, 320, 1, 1],
		]

		# building first layer
		assert input_size % 32 == 0
		input_channel = int(input_channel * width_mult)
		self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
		self.features = [conv_bn(3, input_channel, 2)]
		# building inverted residual blocks
		for t, c, n, s in interverted_residual_setting:
			output_channel = int(c * width_mult)
			for i in range(n):
				if i == 0:
					self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
				else:
					self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
				input_channel = output_channel
		# building last several layers
		self.features.append(conv_1x1_bn(input_channel, self.last_channel))
		# make it nn.Sequential
		self.features = nn.Sequential(*self.features)

		# building classifier
		self.fc = nn.Sequential(
			nn.Dropout(0.2),
			nn.Linear(self.last_channel, n_class),
		)

		self._initialize_weights()

	def forward(self, x):
		x = self.features(x)
		x = x.mean(3).mean(2)
		x = self.fc(x)
		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

	def load_pretrained_imagenet(self, model_file):
		pretrained_dict = torch.load(model_file)
		model_dict = self.state_dict()
		update_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
		model_dict.update(update_dict)
		self.load_state_dict(model_dict)

	def load(self, model_file):
		trained_dict = torch.load(model_file)["model"]
		model_dict = self.state_dict()
		model_dict.update(trained_dict)
		self.load_state_dict(model_dict)


#------------------------------------------------------------------------------
#	Data preprocessing
#------------------------------------------------------------------------------
class ImageFolderLoader(object):
	def __init__(self, dir_image, color_channel="RGB",
				batch_size=32, n_workers=1, pin_memory=True, shuffle=True):

		# Storage parameters
		super(ImageFolderLoader, self).__init__()
		self.dir_image = dir_image
		self.color_channel = color_channel
		self.batch_size = batch_size
		self.n_workers = n_workers
		self.pin_memory = pin_memory
		self.shuffle = shuffle

		self.fnc_CvtChannel = lambda img_PIL: img_PIL.convert(self.color_channel)
		self.mean = [0.485, 0.456, 0.406]
		self.std = [0.229, 0.224, 0.225]

	@property
	def transform_train(self):
		return torchvision.transforms.Compose([
			torchvision.transforms.Lambda(self.fnc_CvtChannel),
			torchvision.transforms.RandomHorizontalFlip(),
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=self.mean, std=self.std)
		])

	@property
	def transform_valid(self):
		return torchvision.transforms.Compose([
			torchvision.transforms.Lambda(self.fnc_CvtChannel),
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=self.mean, std=self.std)
		])

	@property
	def dataset_train(self):
		return ImageFolder(self.dir_image, transform=self.transform_train)

	@property
	def dataset_valid(self):
		return ImageFolder(self.dir_image, transform=self.transform_valid)

	@property
	def train_loader(self):
		return DataLoader(self.dataset_train,
			batch_size=self.batch_size,
			num_workers=self.n_workers,
			pin_memory=self.pin_memory,
			shuffle=self.shuffle,
		)

	@property
	def valid_loader(self):
		return DataLoader(self.dataset_valid,
			batch_size=self.batch_size,
			num_workers=self.n_workers,
			pin_memory=self.pin_memory,
			shuffle=self.shuffle,
		)