#------------------------------------------------------------------------------
#	Import
#------------------------------------------------------------------------------
import torch, os, math
import numpy as np
import torch.nn.functional as F

from libs import image


#------------------------------------------------------------------------------
#	Function to split list of files into chunks of files
#------------------------------------------------------------------------------
def split_chunks(files, n_chunks):
	n_files = len(files)
	samples_per_chunk = int(n_files / n_chunks)

	chunks = []
	counter = 0
	for i in range(n_chunks-1):
		chunks.append(files[counter:counter+samples_per_chunk])
		counter += samples_per_chunk
	chunks.append(files[counter:])
	return chunks


#------------------------------------------------------------------------------
#	Pool to read, crop patches and send to queue
#------------------------------------------------------------------------------
def pool_readcrop_patches(process_idx, files, patch_sz, stride, color_channel,
		queue_patches):

	# Initialize
	print("[Process-{} pool_readcrop_patches] Started with {} samples".format(
		process_idx, len(files)
	))

	# Loop over provided files
	for file in files:
		# Get patches of an image
		img = image.read(file, color_channel)
		coords, _, _ = image.slide2d(
			sz=img.shape[:2],
			K=patch_sz,
			S=stride,
		)
		patches = image.crop_patches(
			img=img,
			coords=coords,
			patch_sz=patch_sz,
		)

		# Preprocess cropped patches
		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]
		patches = torch.tensor(
			data=patches/255,
			dtype=torch.float32,
		)
		for c in range(3):
			patches[:, c, :, :] = (patches[:, c, :, :] - mean[c]) / std[c]

		# Put item to queue_patches
		fname = file.split("/")[-1]
		label = file.split("/")[-2].lower()
		item = (fname, label, patches, coords)
		while queue_patches.full():
			pass
		queue_patches.put(item)

	# Finish the process
	# queue_patches.close()
	print("[Process-{} pool_readcrop_patches] Finished".format(process_idx))
	while not queue_patches.empty():
		pass


#------------------------------------------------------------------------------
#	Pool to predict labels for patches
#------------------------------------------------------------------------------
def pool_predict_patches(process_idx, model, queue_patches, queue_labels,
		out_dir=None):

	# Initialize
	print("[Process-{} pool_predict_patches] Started".format(
		process_idx
	))
	device = model.params["device"]
	model = model.to(device)
	model.eval()
	items = []

	# Loop forever until an outside process kill this process
	while True:
		# Check whether queue has item to get
		if not queue_patches.empty():
			# Get items
			item = queue_patches.get()
			if (type(item) is str) and (item=="exit"):
				break
			fname, patches, coords = item

			# Predict
			inputs = torch.tensor(patches, dtype=torch.float32, device=device)
			inputs = 2*(inputs/255 - 0.5)
			logits = model(inputs)
			_, preds = torch.max(logits.data, dim=1)
			labels = preds.cpu().numpy()

			# Store
			if out_dir is None:
				items.append((fname, labels, coords))
			else:
				save_dict = {"labels":labels, "coords":coords}
				np.save(os.path.join(out_dir, fname), save_dict)

	# Finish the process
	if out_dir is None:
		queue_labels.put(items)
	queue_patches.close()
	queue_labels.close()
	print("[Process-{} pool_predict_patches] Finished".format(process_idx))


#------------------------------------------------------------------------------
#	Pool to predict labels for patches
#------------------------------------------------------------------------------
def pool_predict_softmax_patches(process_idx, model, device, queue_patches,
		queue_labels, max_samples, out_dir=None):

	# Initialize
	print("[Process-{} pool_predict_softmax_patches] Started".format(
		process_idx
	))
	model = model.to(device)
	model.eval()
	items = []

	# Loop forever until an outside process send signal to kill this process
	while True:
		# Check whether queue has item to get
		if not queue_patches.empty():
			# Get items
			item = queue_patches.get()
			if (type(item) is str) and (item=="exit"):
				break
			fname, label, patches, coords = item

			# Split patches into chunks
			n_samples = patches.shape[0]
			chunks = []

			if n_samples > max_samples:
				counter = 0
				n_chunks = int(math.ceil(n_samples / max_samples))
				for i in range(n_chunks-1):
					chunks.append(list(range(counter, counter+max_samples)))
					counter += max_samples
				chunks.append(list(range(counter, n_samples)))

			else:
				chunks.append(list(range(0, n_samples)))

			# Predict softmax
			softmaxs = []
			for chunk in chunks:
				inputs = patches[chunk,...].to(device)
				logits = model(inputs)
				softmax = F.softmax(logits, dim=1).detach().cpu().numpy()
				softmaxs.append(softmax)
			softmaxs = np.concatenate(softmaxs, axis=0)

			# Store
			if out_dir is None:
				items.append((fname, softmaxs, coords))
			else:
				dst_dir = os.path.join(out_dir, label)
				save_dict = {"softmaxs":softmaxs, "coords":coords}
				np.save(os.path.join(dst_dir, fname), save_dict)

	# Finish the process
	if out_dir is None:
		queue_labels.put(items)
	queue_patches.close()
	queue_labels.close()
	print("[Process-{} pool_predict_patches] Finished".format(process_idx))