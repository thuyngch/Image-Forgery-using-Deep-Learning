#------------------------------------------------------------------------------
#	Import
#------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


#------------------------------------------------------------------------------
#	Convolutional Autoencoder
#------------------------------------------------------------------------------
class ConvAutoencoder(nn.Module):
	def __init__(self):
		super(ConvAutoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 8, 3, stride=1, padding=1),
			nn.LeakyReLU(inplace=True),
			nn.MaxPool2d(2, stride=2),

			nn.Conv2d(8, 16, 3, stride=1, padding=1),
			nn.LeakyReLU(inplace=True),
			nn.MaxPool2d(2, stride=2),

			nn.Conv2d(16, 32, 3, stride=1, padding=1),
			nn.LeakyReLU(inplace=True),
			nn.MaxPool2d(2, stride=2),

			nn.Conv2d(32, 64, 3, stride=1, padding=1),
			nn.LeakyReLU(inplace=True),
			nn.MaxPool2d(2, stride=2)
		)
		self.decoder = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
			nn.LeakyReLU(inplace=True),

			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1),
			nn.LeakyReLU(inplace=True),

			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.ConvTranspose2d(16, 8, 3, stride=1, padding=1),
			nn.LeakyReLU(inplace=True),

			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.ConvTranspose2d(8, 3, 3, stride=1, padding=1),
			nn.Tanh()
		)

	def forward(self, X):
		encoded = self.encoder(X)
		decoded = self.decoder(encoded)
		return decoded

	def load_model(self, model_file):
		trained_dict = torch.load(model_file)
		self.load_state_dict(trained_dict)

	def summary(self, input_shape):
		"""input_shape: (C, H, W)"""
		summary(self, input_size=input_shape)


#------------------------------------------------------------------------------
#	Normalized loss for training the Convolutional AutoEncoder
#------------------------------------------------------------------------------
def loss_norm_fn(outputs, targets, labels, loss_weights):
	N = outputs.size()[0]
	loss = (outputs - targets) ** 2
	loss = loss.view(N,-1)
	w_vect = loss_weights[labels]
	loss = torch.mean(loss, 1) * w_vect
	return torch.mean(loss)


#------------------------------------------------------------------------------
#	Convolutional Classifier
#------------------------------------------------------------------------------
class ConvClassifier(nn.Module):
	def __init__(self, params):
		# Storage parameters
		super(ConvClassifier, self).__init__()
		self.params = params
		self.encoded_sz = int(64 * ((params["patch_sz"]/16)**2))
		self.last_iter = 0
		self.last_epoch = 0

		# Network architecture
		self.encoder = nn.Sequential(					# N x 3 x 32 x 32
			nn.Conv2d(3, 8, 3, stride=1, padding=1),	# N x 8 x 32 x 32
			nn.BatchNorm2d(8),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2),					# N x 8 x 16 x 16

			nn.Conv2d(8, 16, 3, stride=1, padding=1),	# N x 16 x 16 x 16
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2),					# N x 16 x 8 x 8

			nn.Conv2d(16, 32, 3, stride=1, padding=1),	# N x 32 x 8 x 8
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2),					# N x 32 x 4 x 4

			nn.Conv2d(32, 64, 3, stride=1, padding=1),	# N x 64 x 4 x 4
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2)					# N x 64 x 2 x 2
		)
		self.classifier = nn.Sequential(
			# nn.Dropout(p=0.5),
			nn.Linear(self.encoded_sz, 2),				# N x 2
		)

	def initialize(self):
		for name, param in self.named_parameters():
			# print(name, param.shape)
			if (len(param.shape)>2):
				nn.init.xavier_uniform_(param,
					gain=nn.init.calculate_gain("relu"))
			# else:
			# 	param.data.fill_(0.01)

	def load_pretrained(self):
		model_dict = self.state_dict()
		pretrained_dict = torch.load(self.params['model_pretrain_file'])
		pretrained_dict = {k: v for k, v in pretrained_dict.items()
							if k in model_dict}
		model_dict.update(pretrained_dict)
		self.load_state_dict(model_dict)

	def load_finetuned(self):
		model_dict = self.state_dict()
		finetuned_dict = torch.load(self.params['model_finetune_file'])
		model_dict.update(finetuned_dict)
		self.load_state_dict(model_dict)

	def forward(self, X):
		encoded_tensor = self.encoder(X)
		encoded_vect = encoded_tensor.view(-1, self.encoded_sz)
		logits = self.classifier(encoded_vect)
		return logits

	def summary(self):
		"""input_size: (C, H, W)"""
		input_size = (3, self.params["patch_sz"], self.params["patch_sz"])
		summary(self, input_size=input_size)



#------------------------------------------------------------------------------
#	MobilenetV2
#------------------------------------------------------------------------------
class Block(nn.Module):
	def __init__(self, in_planes, out_planes, expansion, stride):
		super(Block, self).__init__()
		self.stride = stride

		planes = expansion * in_planes
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn3 = nn.BatchNorm2d(out_planes)

		self.shortcut = nn.Sequential()
		if stride == 1 and in_planes != out_planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(out_planes),
			)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))
		out = out + self.shortcut(x) if self.stride==1 else out
		return out


class MobileNetV2(nn.Module):
	# (expansion, out_planes, num_blocks, stride)
	cfg = [(1,  16, 1, 1),
		   (6,  24, 2, 1),
		   (6,  32, 3, 2),
		   (6,  64, 4, 2),
		   (6,  96, 3, 1),
		   (6, 160, 3, 2),
		   (6, 320, 1, 1)]

	def __init__(self, n_classes=2):
		super(MobileNetV2, self).__init__()
		self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(32)
		self.layers = self._make_layers(in_planes=32)
		self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn2 = nn.BatchNorm2d(1280)
		self.linear = nn.Linear(1280, n_classes)

	def _make_layers(self, in_planes):
		layers = []
		for expansion, out_planes, num_blocks, stride in self.cfg:
			strides = [stride] + [1]*(num_blocks-1)
			for stride in strides:
				layers.append(Block(in_planes, out_planes, expansion, stride))
				in_planes = out_planes
		return nn.Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layers(out)
		out = F.relu(self.bn2(self.conv2(out)))
		out = F.avg_pool2d(out, 8)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out

	def load(self, model_file):
		trained_dict = torch.load(model_file)
		self.load_state_dict(trained_dict)