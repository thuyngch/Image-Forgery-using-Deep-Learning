#------------------------------------------------------------------------------
#	Import
#------------------------------------------------------------------------------
import cv2
import numpy as np
from random import shuffle
from matplotlib import pyplot as plt

import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os, sys, json
from tqdm import tqdm
from itertools import repeat
from libs import image


#------------------------------------------------------------------------------
#	Dataset and Dataloader of subsets' patches
#------------------------------------------------------------------------------
class SubsetPatchesDataset(Dataset):

	def __init__(self, subsets_file, ind_subsets,
						patch_sz, stride, transform=None):
		# Storage parameters
		super(SubsetPatchesDataset, self).__init__()
		self.subsets_file = subsets_file
		self.ind_subsets = ind_subsets
		self.patch_sz = patch_sz
		self.stride = stride
		self.transform = transform

		# Get list of files to images
		self.files = get_files_from_subsets(self.subsets_file,
											self.ind_subsets)

	def __len__(self):
		return len(self.list_files)

	def __getitem__(self, idx):
		file = self.files[idx]
		img = image.read(file)
		coords, _, _ = image.slide2d(img.shape[:2], self.patch_sz, self.stride)
		patches = image.patches(img, coords, self.patch_sz)
		if self.transform:
			patches = self.transform(patches)
		return patches


class SubsetPatchesLoader(object):
	def __init__(self, subsets_file, ind_subsets, patch_sz, stride,
						color_channel, batch_size, n_workers,
						pin_memory=True, shuffle=True):
		# Storage parameters
		super(SubsetPatchesLoader, self).__init__()
		self.subsets_file = subsets_file
		self.ind_subsets = ind_subsets
		self.patch_sz = patch_sz
		self.stride = stride
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
			torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

	@property
	def dataset(self):
		return SubsetPatchesDataset(
						subsets_file=self.subsets_file,
						ind_subsets=self.ind_subsets,
						patch_sz=self.patch_sz,
						stride=self.stride,
						transform=self.transform)

	@property
	def loader(self):
		return DataLoader(self.dataset,
						batch_size=self.batch_size,
						num_workers=self.n_workers,
						pin_memory=self.pin_memory,
						shuffle=self.shuffle)


#------------------------------------------------------------------------------
#	Get list of files from subsets
#------------------------------------------------------------------------------
def get_files_from_subsets(subsets_file, ind_subsets):
	with open(subsets_file, "r") as fp:
		subsets = json.load(fp)

	list_files = []
	for i in ind_subsets:
		files = subsets[str(i)]
		list_files += files

	return list_files


#------------------------------------------------------------------------------
#	Get patches' information from raw file
#------------------------------------------------------------------------------
def get_patches_info_from_file(patches_info_file):
	with open(patches_info_file, "r") as fp:
		content = fp.read()
		lines = content.split("\n")
		patch_coords = [line.split(", ") for line in lines if len(line)>1]
		files, Ys, Xs = tuple(zip(*patch_coords))

	return files, Ys, Xs


#------------------------------------------------------------------------------
#	Get patches' information
#------------------------------------------------------------------------------
def get_patches_info(subsets_file, patches_info_file, test_subset, folder):
	# Get list of files from subsets
	ind_train = list(range(6))
	ind_train.remove(test_subset)
	train_files = get_files_from_subsets(subsets_file, ind_train)

	# Get patches' information
	files, Ys, Xs = get_patches_info_from_file(patches_info_file)
	files = [os.path.join(folder, file) for file in files]

	# Verify the matching
	patches_info = []
	for i, file in enumerate(files):
		y, x = int(Ys[i]), int(Xs[i])
		file_name = file.split("/")[-1]
		if file_name in train_files:
			patches_info.append((file, y, x))
	return patches_info


#------------------------------------------------------------------------------
#   Split training and validating sets
#------------------------------------------------------------------------------
def split_train_eval(data_patches, ratio):
	shuffle(data_patches)
	n_samples = len(data_patches)
	n_train = int(n_samples*ratio)
	train_patches = data_patches[:n_train]
	valid_patches = data_patches[n_train:]
	return train_patches, valid_patches


#------------------------------------------------------------------------------
#	Rotate image anti-clockwise
#------------------------------------------------------------------------------
def rotate(img, times):
	img_rot = np.rot90(img, k=times, axes=(0,1))
	return img_rot


#------------------------------------------------------------------------------
#   Read, crop, and save patch
#------------------------------------------------------------------------------
def pool_crop_patch(args):
	"""
	[src_file] File of the source image.
	[y, x] Coordinate of the left-top corner of the patch.
	[patch_sz] Size of patch.
	[out_dir] Output directory to write patch image.
	[prefix] Prefix of the filename of the patch.
	[idx] Index of output file.
	"""
	# Unroll arguments
	(src_file, y, x), patch_sz, out_dir, prefix, idx = args

	# Read, crop patch, and save
	i = y + 16 - int(patch_sz/2)
	j = x + 16 - int(patch_sz/2)
	img_src = image.read(src_file, channel="RGB")
	if (i >=0) and (j>=0) and (i+patch_sz<=img_src.shape[0]) and (j+patch_sz<=img_src.shape[1]):
		patch = img_src[i:i+patch_sz, j:j+patch_sz, :]
		dst_file = os.path.join(out_dir, "%s_%d.png" % (prefix, idx))
		image.write(dst_file, patch)


def crop_and_save(pools, data_patches, patch_sz, out_dir, prefix):
	n_samples = len(data_patches)
	args = zip(data_patches, repeat(patch_sz),
				repeat(out_dir), repeat(prefix), range(n_samples))
	for _ in tqdm(pools.imap_unordered(pool_crop_patch, args), total=n_samples):
		pass


#------------------------------------------------------------------------------
#	Create negative patches based on number of positive patches
#------------------------------------------------------------------------------
def create_neg_based_on_pos(N_pos, fnames, out_file, au_dir):
	# Some additional functions
	def gen_patch_corners(size, num):
		h,w,c = size; h -= 32; w -=32
		x = np.random.randint(0, w, size=(num,1))
		y = np.random.randint(0, h, size=(num,1))
		corner = np.hstack((y,x))
		return corner.astype(int)
	def write_to_file(fp, file, corners):
		for (y, x) in corners:
			fp.write("%s, %d, %d\n" % (file, y, x))

	# Create negative patches
	files = [os.path.join(au_dir, fname) for fname in fnames]
	N_neg = int(N_pos / len(files)) + 1

	fp = open(out_file, "w")
	for i, file in enumerate(files):
		img = cv2.imread(file)
		corner = gen_patch_corners(img.shape, N_neg)
		write_to_file(fp, fnames[i], corner)
	fp.close()


#------------------------------------------------------------------------------
#	Visualize some patches
#------------------------------------------------------------------------------
def visualize_patches(patches, H, W):
	# Checking
	N = len(patches)
	if N < H*W:
		sys.exit("Number of patches is smaller than desired size")

	# Select patches
	idx = np.random.randint(0, N, size=(H*W)).astype(int)
	patches = np.array(patches)
	patches_selected = patches[idx, ...]

	# Plot
	plt.figure(1)
	for i in range(H):
		for j in range(W):
			ind = i*W+j; patch = patches_selected[ind]
			plt.subplot(H, W, ind+1); plt.imshow(patch); plt.axis("off")
	plt.show()