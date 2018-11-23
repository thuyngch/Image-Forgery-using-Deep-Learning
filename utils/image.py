#------------------------------------------------------------------------------
#	Import
#------------------------------------------------------------------------------
import numpy as np
from PIL import Image
from itertools import repeat


#------------------------------------------------------------------------------
#	Read/Write image
#------------------------------------------------------------------------------
def read(file, channel="RGB"):
	""" Return an ndarray image of shape (W, H, C)"""
	return np.array(Image.open(file).convert(channel))

def write(file, img):
	""" img is an ndarray matrix of shape (W, H, C)"""
	Image.fromarray(img).save(file)


#------------------------------------------------------------------------------
#	Get coordinates of patches when sliding 2D
#------------------------------------------------------------------------------
def slide2d(sz, K, S):
	H, W = sz
	i = 0; j = 0
	n_H, n_W = 0, 0
	coords = []
	while True:
		if i+K > H:
			break
		n_W = 0
		while True:
			if j+K > W:
				break
			coords.append((i, j))
			j += S
			n_W += 1
		i += S
		j = 0
		n_H += 1

	return coords, n_H, n_W


#------------------------------------------------------------------------------
#	Crop patches of an images
#------------------------------------------------------------------------------
def crop_patches(img, coords, patch_sz):
	def crop(img, coord, patch_sz):
		i, j = coord
		patch = img[i:i+patch_sz, j:j+patch_sz, ...].transpose((2,0,1))
		return patch

	patch_obj = map(crop, repeat(img), coords, repeat(patch_sz))
	patches = list(patch_obj)
	return np.array(patches)


#------------------------------------------------------------------------------
#	Get coordinates of up-left corners of surrounding patches of a patch
#------------------------------------------------------------------------------
def surround(ith_p, coords, sur, S):
	# Propose some surrounding coordinates
	i, j = coords[ith_p]
	if sur==8:
		c_sur_raw = [
					(i-S,j-S), (i-S,j), (i-S,j+S),
					(i,j-S), (i,j+S),
					(i+S,j-S), (i+S,j), (i+S,j+S)]
	elif sur==24:
		S2 = 2*S
		c_sur_raw = [
				(i-S2,j-S2), (i-S2,j-S), (i-S2,j), (i-S2,j+S), (i-S2,j+S2),
				(i-S,j-S2), (i-S,j-S), (i-S,j), (i-S,j+S), (i-S,j+S2),
				(i,j-S2), (i,j-S), (i,j+S), (i,j+S2),
				(i+S,j-S2), (i+S,j-S), (i+S,j), (i+S,j+S), (i+S,j+S2),
				(i+S2,j-S2), (i+S2,j-S), (i+S2,j), (i+S2,j+S), (i+S2,j+S2)]
	# Eliminate invalid coordinates
	i_max, j_max = np.max(coords, axis=0)
	c_sur = [(i,j) for (i,j) in c_sur_raw if 0<=i<=i_max and 0<=j<=j_max]
	# Get indeces of valid coordinates
	coords_list = list(map(tuple, coords))
	ith_sur = [coords_list.index(c) for c in c_sur]
	return ith_sur


#------------------------------------------------------------------------------
#	Post-processing
#------------------------------------------------------------------------------
def post(ith_p, softmaxs, coords, sur, thres, stride):
	softmax_p = softmaxs[ith_p]
	ith_sur = surround(ith_p, coords, sur, stride); N = len(ith_sur)
	softmax_sur = softmaxs[ith_sur]; sum_sur = np.sum(softmax_sur)
	prob = (sum_sur + softmax_p) / (N + 1)
	val = 1 if prob>=thres else 0
	return val

def post_process(softmaxs, coords, sur, thres, stride, pools=None):
	N = len(softmaxs)
	if pools is not None:
		args = zip(range(N), repeat(softmaxs), repeat(coords),
						repeat(sur), repeat(thres), repeat(stride))
		labels = pools.starmap(post, args)
	else:
		obj = map(post, range(N), repeat(softmaxs), repeat(coords),
						repeat(sur), repeat(thres), repeat(stride))
		labels = list(obj)

	return labels


#------------------------------------------------------------------------------
#	Fusion
#------------------------------------------------------------------------------
def fusion(labels):
	val_sum = np.sum(labels)
	return 1 if val_sum>=1 else 0


#------------------------------------------------------------------------------
#	Reconstruct binary map
#------------------------------------------------------------------------------
def reconstruct_binmap(label_vect, coords, img_shape, patch_sz):
	img_recons = np.zeros(img_shape[:2], dtype=np.uint8)
	
	p_one = 255*np.ones([patch_sz, patch_sz], dtype=np.uint8)
	p_zero = np.zeros([patch_sz,patch_sz], dtype=np.uint8)

	for i in range(len(coords)):
		label = label_vect[i]
		i, j = coords[i]
		img_recons[i:i+patch_sz, j:j+patch_sz] = p_one if label else p_zero
	return img_recons


#------------------------------------------------------------------------------
#	Reconstruct heat map
#------------------------------------------------------------------------------
# def reconstruct_heatmap(softmaxs, coords, img_shape, patch_sz):
# 	img_recons = np.zeros(img_shape[:2], dtype=np.uint8)
# 	p_one = np.ones([patch_sz, patch_sz], dtype=np.uint8)

# 	for idx in range(len(coords)):
# 		softmax = int(softmaxs[idx]*255) * p_one
# 		i, j = coords[idx]
# 		patch = img_recons[i:i+patch_sz, j:j+patch_sz]
# 		map_f, map_i = (patch==0).astype(int), (patch!=0).astype(int)
# 		content = softmax*map_f + 0.5*(softmax+patch)*map_i
# 		img_recons[i:i+patch_sz, j:j+patch_sz] = content.astype(np.uint8)
# 	return img_recons

def reconstruct_heatmap(softmaxs, coords, img_shape, patch_sz):
	img_recons = np.zeros(img_shape[:2], dtype=np.uint8)
	p_one = np.ones([patch_sz, patch_sz], dtype=np.uint8)

	for idx in range(len(coords)):
		i, j = coords[idx]
		softmax = int(255*softmaxs[idx]) * p_one
		patch = img_recons[i:i+patch_sz, j:j+patch_sz]
		content = 0.5*(softmax+patch)
		img_recons[i:i+patch_sz, j:j+patch_sz] = content.astype(np.uint8)
	return img_recons


#------------------------------------------------------------------------------
#	Data normalize
#------------------------------------------------------------------------------
def normalize(X, mean, std):
	n_channels = len(mean)
	for c in range(n_channels):
		X[c,...] = (X[c,...] - mean[c]) / std[c]
	return X