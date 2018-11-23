#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import torch, math
import torch.nn as nn
import torch.nn.functional as F


#------------------------------------------------------------------------------
#	Bottleneck
#------------------------------------------------------------------------------
class Bottleneck(nn.Module):
	def __init__(self, num_inp_channels, num_hidden_channels, stride):
		super(Bottleneck, self).__init__()
		self.conv1x1_down = nn.Sequential(
			nn.Conv2d(
				in_channels=num_inp_channels,
				out_channels=num_hidden_channels,
				kernel_size=1,
				stride=1,
				bias=False,
			),
			nn.BatchNorm2d(num_features=num_hidden_channels),
			nn.ReLU(inplace=True),
		)
		self.conv3x3 = nn.Sequential(
			nn.Conv2d(
				in_channels=num_hidden_channels,
				out_channels=num_hidden_channels,
				kernel_size=3,
				stride=stride,
				padding=1,
				bias=False,
			),
			nn.BatchNorm2d(num_features=num_hidden_channels),
			nn.ReLU(inplace=True),
		)
		self.conv1x1_up = nn.Sequential(
			nn.Conv2d(
				in_channels=num_hidden_channels,
				out_channels=4*num_hidden_channels,
				kernel_size=1,
				stride=1,
				bias=False,
			),
			nn.BatchNorm2d(num_features=4*num_hidden_channels),
		)
		if (stride!=1) or (num_inp_channels!=4*num_hidden_channels):
			self.skip_conn = nn.Sequential(
				nn.Conv2d(
					in_channels=num_inp_channels,
					out_channels=4*num_hidden_channels,
					kernel_size=1,
					stride=stride,
					bias=False,
				),
				nn.BatchNorm2d(num_features=4*num_hidden_channels),
			)
		else:
			self.skip_conn = nn.Sequential()


	def forward(self, X):
		out1 = self.conv1x1_down(X)
		out2 = self.conv3x3(out1)
		out3 = self.conv1x1_up(out2)
		out4 = out3 + self.skip_conn(X)
		out = F.relu(out4, inplace=True)
		return out


#------------------------------------------------------------------------------
#	General ResNet
#------------------------------------------------------------------------------
class ResNet(nn.Module):
	def __init__(self, input_size, list_of_bottlenecks, num_classes=2):
		super(ResNet, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(
				in_channels=3,
				out_channels=64,
				kernel_size=1,
				stride=2,
				bias=False,
			),
			nn.BatchNorm2d(num_features=64),
			nn.ReLU(inplace=True),
		)
		self.block1 = self.resnet_block(
			num_inp_channels=64,
			num_hidden_channels=64,
			num_bottlenecks=list_of_bottlenecks[0],
			stride=2,
		)
		self.block2 = self.resnet_block(
			num_inp_channels=256,
			num_hidden_channels=128,
			num_bottlenecks=list_of_bottlenecks[1],
			stride=2,
		)
		self.block3 = self.resnet_block(
			num_inp_channels=512,
			num_hidden_channels=256,
			num_bottlenecks=list_of_bottlenecks[2],
			stride=2,
		)
		self.block4 = self.resnet_block(
			num_inp_channels=1024,
			num_hidden_channels=512,
			num_bottlenecks=list_of_bottlenecks[3],
			stride=2,
		)
		self.global_avg_pool = nn.AvgPool2d(kernel_size=int(input_size/32))
		self.fc = nn.Linear(2048, num_classes)


	def resnet_block(self, num_inp_channels, num_hidden_channels, num_bottlenecks, stride):
		layers = []
		strides = [stride] + [1]*(num_bottlenecks-1)
		for stride in strides:
			layer = Bottleneck(
				num_inp_channels=num_inp_channels,
				num_hidden_channels=num_hidden_channels,
				stride=stride,
			)
			layers.append(layer)
			num_inp_channels = 4*num_hidden_channels

		return nn.Sequential(*layers)


	def forward(self, X):
		conv1 = self.conv1(X)
		block1 = self.block1(conv1)
		block2 = self.block2(block1)
		block3 = self.block3(block2)
		block4 = self.block4(block3)
		global_avg_pool = self.global_avg_pool(block4).view(X.shape[0], -1)
		logits = self.fc(global_avg_pool)
		return logits


	def initialize_weights(self):
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


	def load(self, model_file):
		trained_dict = torch.load(model_file)
		self.load_state_dict(trained_dict)


#------------------------------------------------------------------------------
#	Specific Resnet
#------------------------------------------------------------------------------
def ResNet50(input_size):
	return ResNet(
		input_size=input_size,
		list_of_bottlenecks=[3,4,6,3],
		num_classes=2,
	)


def ResNet101(input_size):
	return ResNet(
		input_size=input_size,
		list_of_bottlenecks=[3,4,23,3],
		num_classes=2,
	)


def ResNet150(input_size):
	return ResNet(
		input_size=input_size,
		list_of_bottlenecks=[3,8,36,3],
		num_classes=2,
	)