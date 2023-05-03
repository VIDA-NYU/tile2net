import logging
import os
import glob
import pathlib
import numpy as np
from PIL import Image
from skimage.io import imsave


def image_splitter(src_path: str) -> None:
	"""
	Splits the side by side segmention results to image and annotations
	Parameters
	----------
	src_path: str
		the path to the source files with *.png format
	"""
	imgs = glob.glob(src_path)
	img_dest = os.path.join(pathlib.PurePath(src_path).parent, 'splitted', 'images')
	if not os.path.exists(img_dest):
		os.makedirs(img_dest)
	annot_dest = os.path.join(pathlib.PurePath(src_path).parent, 'splitted', 'annotations')
	if not os.path.exists(annot_dest):
		os.makedirs(annot_dest)
	for im in imgs:
		name = pathlib.PurePath(im).name
		img = np.array(Image.open(im))
		lable = img[:, int(img.shape[1]/2): img.shape[1], :]
		phot = img[:, 0:int(img.shape[1]/2), :]
		phimg = Image.fromarray(phot)
		phimg.save(os.path.join(img_dest, name))
		# mask
		labimg = Image.fromarray(lable)
		labimg.save(os.path.join(annot_dest, name))


classes = {'sw': [0, 0, 255], 'road': [0, 128, 0], 'crosswalk': [255, 0, 0], 'background': [0, 0, 0]}


def direct2gray(src_path: str, classes: dict) -> None:
	"""
	Creates one channel gray annotation masks from the colored masks
	Parameters
	----------
	src_path: str
		the path to the colored annotation files, it should be  ./../*.png
	classes: dict
		dictionary of different materials and their RGB values
	"""
	dest = os.path.join(pathlib.PurePath(src_path).parent, 'gray')
	if not os.path.exists(dest):
		os.makedirs(dest)
	for img_file in glob.glob(src_path):
		img = Image.open(img_file).convert('RGB')
		data = np.array(img)
		mask = np.zeros(data.shape)
		for idx, (c, v) in enumerate(classes.items(), 1):
			mask = np.all(data == classes[c], axis=-1)*idx if idx == 1 else mask + np.all(
				data == classes[c],
				axis=-1
			)*idx
		imsave(os.path.join(dest, pathlib.PurePath(img_file).name), mask.astype(np.uint8))


def overlay(img_path, label_path, alpha, sbs=None):
	"""
	Overlays the segmentation mask on the original image.
	Parameters
	----------
	img_path: str
		Path to the original image
	label_path: str
		Path to the segmentation mask
	alpha: float
		Transparency of the segmentation mask
	sbs: bool
		Whether the image is side by side or not

	Returns
	-------
	None
	saves the overlayed image in the same directory as the original image
	"""
	images = glob.glob(img_path)
	dest = os.path.join(pathlib.PurePath(label_path).parent, 'overlay')
	if not os.path.exists(dest):
		os.makedirs(dest)

	for img in images:
		img_name = pathlib.PurePath(img).name
		label = os.path.join(label_path, img_name)
		img_np = Image.open(img)
		try:
			if sbs:
				im = np.array(img_np)
				lable = im[:, int(im.shape[1]/2): im.shape[1], :]
				phot = im[:, 0:int(im.shape[1]/2), :]
				img_np = Image.fromarray(phot)
				lbl_np = Image.fromarray(lable)
			else:
				lbl_np = Image.open(label)

			lbl_np = lbl_np.convert('RGB')
			composited = Image.blend(img_np, lbl_np, alpha)
			composited_fn = 'composited_{}.png'.format(img_name)
			composited_fn = os.path.join(dest, composited_fn)
			composited.save(composited_fn)
			print(lbl_np)

		except FileNotFoundError:
			print(f'annotation {img_name} not found')
			continue


def sbs(src_path, label_path):
	"""
	Combine images and labels side by side
	Parameters
	----------
	src_path: str
		path to the images
	label_path: str
		path to the labels
	"""
	images = glob.glob(src_path)
	dest = os.path.join(pathlib.PurePath(src_path).parent, 'side-by-side')
	if not os.path.exists(dest):
		os.makedirs(dest)

	for img in images:

		img_name = pathlib.PurePath(img).name
		label = os.path.join(label_path, img_name)
		print(label)
		img_np = Image.open(img)
		try:
			lbl_np = Image.open(label)
			print(lbl_np)
			dst = Image.new('RGB', (img_np.width + lbl_np.width, img_np.height))
			dst.paste(img_np, (0, 0))
			dst.paste(lbl_np, (img_np.width, 0))
			dst.save(os.path.join(dest, img_name))

		except FileNotFoundError:
			print(f'annotation {img_name} not found')
			continue


def fill_colormap():
	"""
	Fill colormap with appropriate values corresponding to the classes
	for the segmentation mask of Tile2Net
	Returns
	-------
	palette : list
		Colormap for the segmentation mask
	"""
	palette = [0, 0, 0,
	           0, 0, 255,
	           0, 128, 0,
	           255, 0, 0]
	zero_pad = 256*3 - len(palette)
	for i in range(zero_pad):
		palette.append(0)
	return palette


def colorize_mask(src_dir):
	"""
	Colorize gray segmentation masks
	"""
	folds = src_dir
	ims_f = glob.glob(folds)
	dest = os.path.join(pathlib.PurePath(src_dir).parent, 'colorized')
	if not os.path.exists(dest):
		os.makedirs(dest)
	for im in ims_f:
		img = Image.open(im)
		ims = np.array(np.array(img))
		new_mask = Image.fromarray(ims.astype(np.uint8)).convert('P')
		new_mask.putpalette(fill_colormap())
		# imgs = np.array(new_mask)
		name = pathlib.PurePath(im).name
		new_mask.save(os.path.join(dest, name))


def im_has_alpha(img_arr):
	"""
	Checks if image has alpha channel
	Parameters
	----------
	img_arr: np.array
		Image array

	Returns
	-------
	True if image has alpha channel, False otherwise
	"""
	h, w, c = img_arr.shape
	return True if c==4 else False


def has_transparency(img):
	"""
	Checks if image has transparency
	Parameters
	----------
	img : PIL Image
		Image to check

	Returns
	-------
	True if image has transparency, False otherwise

	"""
	if img.info.get("transparency", None) is not None:
		return True
	if img.mode=="P":
		transparent = img.info.get("transparency", -1)
		for _, index in img.getcolors():
			if index==transparent:
				return True
	elif img.mode=="RGBA":
		extrema = img.getextrema()
		if extrema[3][0] < 255:
			return True

	return False


def alpha_to_color(imgpath, dest, color=(50, 50, 50)):
	"""Alpha composite an RGBA Image with a specified color.

	Simpler, faster version than the solutions above.

	Source: http://stackoverflow.com/a/9459208/284318

	Keyword Arguments:
	image -- PIL RGBA Image object
	color -- Tuple r, g, b (default 255, 255, 255)

	"""
	for imag in glob.glob(imgpath):
		name = pathlib.PurePath(imag).name
		try:
			image = Image.open(imag)
			image.load()  # needed for split()
			background = Image.new('RGB', image.size, color)
			background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
			if not os.path.exists(dest):
				os.makedirs(dest)
			background.save(os.path.join(dest, name))
		except:
			print(f'{name} is corrupted')
