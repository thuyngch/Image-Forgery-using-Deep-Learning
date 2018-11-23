#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import torch, math
import torch.nn as nn


#------------------------------------------------------------------------------
#	Inverted Residual block
#------------------------------------------------------------------------------
class InvertedResidual(nn.Module):
	"""
	Inverted Residual block of the MobileNet V2.

	[Arguments]
		n_inp_channels : (int) Number of input channels.

		n_out_channels : (int) Number of output channels.

		stride : (int) Stride of convolution.

		expansion : (int) Expansion ratio.
	"""

	def __init__(self, n_inp_channels, n_out_channels, stride, expansion):
		super(InvertedResidual, self).__init__()
		self.stride = stride
		hidden_dim = int(n_inp_channels*expansion)

		# 1x1 Convolution
		self.conv1x1 = nn.Sequential(
			nn.Conv2d(
				in_channels=n_inp_channels,
				out_channels=hidden_dim,
				kernel_size=1,
				stride=1,
				padding=0,
				bias=False,
			),
			nn.BatchNorm2d(num_features=hidden_dim),
			nn.ReLU6(inplace=True),
		)

		# 3x3 depthwise convolution
		self.conv3x3_depthwise = nn.Sequential(
			nn.Conv2d(
				in_channels=hidden_dim,
				out_channels=hidden_dim,
				kernel_size=3,
				stride=stride,
				padding=1,
				groups=hidden_dim,
				bias=False,
			),
			nn.BatchNorm2d(num_features=hidden_dim),
			nn.ReLU6(inplace=True),
		)

		# 1x1 linear convolution
		self.conv1x1_linear = nn.Sequential(
			nn.Conv2d(
				in_channels=hidden_dim,
				out_channels=n_out_channels,
				kernel_size=1,
				stride=1,
				padding=0,
				bias=False,
			),
			nn.BatchNorm2d(num_features=n_out_channels),
		)

		# Skip connection
		if (self.stride==1) and (n_inp_channels!=n_out_channels):
			self.skip_conn = nn.Sequential(
				nn.Conv2d(
					in_channels=n_inp_channels,
					out_channels=n_out_channels,
					kernel_size=1,
					stride=1,
					padding=0,
					bias=False,
				),
				nn.BatchNorm2d(num_features=n_out_channels),
			)
		else:
			self.skip_conn = nn.Sequential()


	def forward(self, x):
		out1 = self.conv1x1(x)
		out2 = self.conv3x3_depthwise(out1)
		out3 = self.conv1x1_linear(out2)
		out = out3 + self.skip_conn(x) if self.stride==1 else out3
		return out


#------------------------------------------------------------------------------
#	MobileNet V2
#------------------------------------------------------------------------------
class MobileNetV2(nn.Module):
	"""
	Model of MobileNet V2.

	[Arguments]
		n_class : (int) Number of classes for the classification problem.

		input_size : (int) Size of the input tensor.
	"""

	def __init__(self, n_class, input_size):
		super(MobileNetV2, self).__init__()
		# The initial layer
		self.conv1 = nn.Sequential(
			nn.Conv2d(
				in_channels=3,
				out_channels=32,
				kernel_size=3,
				stride=2,
				padding=1,
				bias=False,
			),
			nn.BatchNorm2d(num_features=32),
			nn.ReLU6(inplace=True),
		)

		# Bottleneck layers
		sequences = [ # n, out, stride, expansion

			# Original MobileNetV2
			(1, 16, 1, 1),
			(2, 24, 2, 6),
			(3, 32, 2, 6),
			(4, 64, 2, 6),
			(3, 96, 1, 6),
			(3, 160, 2, 6),
			(1, 320, 1, 6),

			# # Modified MobileNetV2
			# (1, 16, 1, 1),
			# (2, 24, 1, 6),
			# (3, 32, 2, 6),
			# (4, 64, 2, 6),
			# (3, 96, 1, 6),
			# (3, 160, 2, 6),
			# (1, 320, 1, 6),
		]
		bottlenecks = []
		inp = 32
		for (n, out, s, e) in sequences:
			strides = [s] + (n-1)*[1]
			for stride in strides:
				bottleneck = InvertedResidual(
					n_inp_channels=inp,
					n_out_channels=out,
					stride=stride,
					expansion=e,
				)
				inp = out
				bottlenecks.append(bottleneck)
		self.bottlenecks = nn.Sequential(*bottlenecks)

		# Convolutional layer -> 1280 channels
		self.conv2 = nn.Sequential(
			nn.Conv2d(
				in_channels=320,
				out_channels=1280,
				kernel_size=1,
				stride=1,
				padding=0,
				bias=False
			),
			nn.BatchNorm2d(num_features=1280),
			nn.ReLU6(inplace=True),
		)

		# Global Average Pooling
		kernel_size = int(input_size / 32)
		self.global_avg_pool = nn.AvgPool2d(kernel_size=kernel_size)

		# Fully connected
		self.fc = nn.Linear(
			in_features=1280,
			out_features=n_class,
			bias=True,
		)

		# Initialization
		self.initialize_weights()


	def forward(self, x):
		out1 = self.conv1(x)
		out2 = self.bottlenecks(out1)
		out3 = self.conv2(out2)
		out4 = self.global_avg_pool(out3).view(-1, 1280)
		out = self.fc(out4)
		return out


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