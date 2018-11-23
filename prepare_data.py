#------------------------------------------------------------------------------
#	Import
#------------------------------------------------------------------------------
import json, argparse, os
from multiprocessing import Pool, cpu_count

from utils import patches
import warnings
warnings.filterwarnings("ignore")


#------------------------------------------------------------------------------
#	Check directories
#------------------------------------------------------------------------------
def check_directories(list_dirs):
	for dir in list_dirs:
		if not os.path.exists(dir):
			print("makedirs", dir)
			os.makedirs(dir)


#------------------------------------------------------------------------------
#	Parse arguments
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
	'--channel', type=str, default="RGB",
	help='Color channel'
)
parser.add_argument(
	'--patch_sz', type=int, default=64,
	help='Patch size'
)
parser.add_argument(
	'--test_subset', type=int, default=5,
	help='Index of the test subset'
)
parser.add_argument(
	'--dir', type=str, default="./data/",
	help='Folder containing the extracted data'
)
parser.add_argument(
	'--casia2_dir', type=str, default="/media/antiaegis/storing/datasets/CASIA2/",
	help='Folder containing CASIA2 database'
)
parser.add_argument(
	'--au_subdir', type=str, default="Au",
	help='Sub-folder containing authentic images'
)
parser.add_argument(
	'--tp_subdir', type=str, default="Tp",
	help='Sub-folder containing tampered images'
)
parser.add_argument(
	'--ratio', type=float, default=0.9,
	help='Training samples ratio'
)
args = parser.parse_args()


#------------------------------------------------------------------------------
#	Initial procedure
#------------------------------------------------------------------------------
# Check directories
DIR = os.path.join(args.dir, "%s_%d_%d" % (
	args.channel, args.patch_sz, args.test_subset
))
TRAIN_AU_DIR = os.path.join(DIR, "train/au")
TRAIN_TP_DIR = os.path.join(DIR, "train/tp")
VALID_AU_DIR = os.path.join(DIR, "valid/au")
VALID_TP_DIR = os.path.join(DIR, "valid/tp")

list_dirs = [
	TRAIN_AU_DIR,
	TRAIN_TP_DIR,
	VALID_AU_DIR,
	VALID_TP_DIR,
]
check_directories(list_dirs)


# Create parallel pools
pools = Pool(processes=cpu_count())


#------------------------------------------------------------------------------
#	Create tampering patches
#------------------------------------------------------------------------------
# Get patches' information
patches_info = patches.get_patches_info(
	subsets_file="dataset/tp_subsets.json",
	patches_info_file="dataset/patch_coord_pos_casiav2.txt",
	test_subset=args.test_subset,
	folder=os.path.join(args.casia2_dir, args.tp_subdir),
)

n_pos_samples = len(patches_info)
print("Number of positive samples:", n_pos_samples)


# Split training and validating set
train_patches, valid_patches = patches.split_train_eval(
	data_patches=patches_info,
	ratio=args.ratio,
)


# Read, crop, and save patches
n_train = len(train_patches)
print("Number of train samples:", n_train)
patches.crop_and_save(
	pools=pools,
	data_patches=train_patches,
	patch_sz=args.patch_sz,
	out_dir=TRAIN_TP_DIR,
	prefix="train_tp",
)

n_valid = len(valid_patches)
print("Number of valid samples:", n_valid)
patches.crop_and_save(
	pools=pools,
	data_patches=valid_patches,
	patch_sz=args.patch_sz,
	out_dir=VALID_TP_DIR,
	prefix="valid_tp",
)


#------------------------------------------------------------------------------
#	Create authentic patches
#------------------------------------------------------------------------------
# Get list of files
with open("dataset/au_subsets.json", "r") as fp:
	au_subsets = json.load(fp)

list_of_train_subsets_ind = list(range(6))
list_of_train_subsets_ind.remove(args.test_subset)

train_files = []
for i in list_of_train_subsets_ind:
	files = au_subsets[str(i)]
	train_files += files


# Generate file containing coordinates of patches in images
patches.create_neg_based_on_pos(
	N_pos=n_pos_samples,
	fnames=train_files,
	out_file="dataset/patch_coord_neg_casiav2.txt",
	au_dir=os.path.join(args.casia2_dir, args.au_subdir),
)


# Get patches' information
patches_info = patches.get_patches_info(
	subsets_file="dataset/au_subsets.json",
	patches_info_file="dataset/patch_coord_neg_casiav2.txt",
	test_subset=args.test_subset,
	folder=os.path.join(args.casia2_dir, args.au_subdir),
)
n_neg_samples = len(patches_info)
print("Number of negative samples:", n_neg_samples)


# Split training and validating set
train_patches, valid_patches = patches.split_train_eval(
	data_patches=patches_info,
	ratio=args.ratio,
)


# Read, crop, and save patches
n_train = len(train_patches)
print("Number of train samples:", n_train)

patches.crop_and_save(
	pools=pools,
	data_patches=train_patches,
	patch_sz=args.patch_sz,
	out_dir=TRAIN_AU_DIR,
	prefix="train_au",
)

n_valid = len(valid_patches)
print("Number of valid samples:", n_valid)

patches.crop_and_save(
	pools=pools,
	data_patches=valid_patches,
	patch_sz=args.patch_sz,
	out_dir=VALID_AU_DIR,
	prefix="valid_au",
)