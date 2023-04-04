import logging
import os
import glob
import math
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import psutil
import random



def deg2num(lat_deg, lon_deg, zoom):
    """
    converts decimal deg lat and long of the tile box to tile numbers

    :param lat_deg:
    :param lon_deg:
    :param zoom:
    :return:
    """

    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile: float = int((lon_deg + 180.0) / 360.0 * n)
    ytile: float = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def num2deg(xtile, ytile, zoom):
    """
    converts tile numbers to lat and long of the tile box
    This returns the NW-corner of the square. Use the function with xtile+1 and/or ytile+1 to get the other corners.
    With xtile+0.5 & ytile+0.5 it will return the center of the tile.
    :param xtile:
    :param ytile:
    :param zoom:
    :return:
    """
    n = 2.0 ** zoom
    lon_deg: float = xtile / n * 360.0 - 180.0
    lat_rad: float = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg: float = math.degrees(lat_rad)
    return lat_deg, lon_deg


def path_check(pathchk):
    filename = os.path.basename(pathchk)
    dir_path = os.path.dirname(pathchk)
    exten = "." + str(filename.split(".")[-1])
    if os.path.exists(pathchk):
        sureinput = str(input(f"ATTENTION! File '{filename}' exists! \nDo you want to OVERWRITE it? [Y/N]"))
        if sureinput.lower() in ['y', 'yes']:
            logging.warning('You will OVERWRITE your old output file!')
            return pathchk
        if sureinput.lower() in ['n', 'no', 'na']:
            newname = str(input('Enter new file name here (without extension)'))
            while os.path.exists(os.path.join(dir_path, newname + exten)):
                newname = str(input("File name exists! Enter new file name here (Without Extension)"))
            pathchk = os.path.join(dir_path, newname + exten)
            logging.info(f"You Changed the output file name to : {os.path.basename(pathchk)}")
            return pathchk
    else:
        return pathchk


def change_name(src_path, *args):
    imgs = glob.glob(src_path)
    dest_path = src_path[:-6] + '/new_name_class/'
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    for c, im in enumerate(imgs):
        img = Image.open(im)
        img.save(dest_path + str(c) + '-' + '.png')


def move_img_ann(img_src, ann_src, ann_dest):
    # ann_dest = os.path.join()
    if not os.path.exists(ann_dest):
        os.makedirs(ann_dest)

    for i in os.listdir(img_src):
        if i in os.listdir(ann_src):
            shutil.copy(os.path.join(ann_src, i), ann_dest)


def move_ann(src, ann_src):
    ann_dest = os.path.join(src, 'removed')
    if not os.path.exists(ann_dest):
        os.makedirs(ann_dest)
    imgs = os.path.join(src, 'images')
    anns = os.path.join(src, 'annotations-rswcw')
    for i in os.listdir(ann_src):
        if i in os.listdir(imgs):
            shutil.move(os.path.join(imgs, i), os.path.join(ann_dest, 'images'))
            shutil.move(os.path.join(anns, i), os.path.join(ann_dest, 'annotations'))


# classes = {'bldg': [255, 0, 0], 'road':[0,128, 0], 'sw': [0, 0, 255], 'background': [0, 0, 0] }

def data_splitter(src_dir, frac_train) -> None:
    ''' This function splits data to train, validation and test sets.

        Args:
            src_dir (str): source directory containing the raw dataset to be splited
            frac_train (float): the fraction dedicated to training

        Returns:
            None
        '''
    img_src_dir = os.path.join(src_dir, 'images', '*.png')
    annot_src_dir = os.path.join(src_dir, 'annotations', '*.png')
    frac2 = (1 + frac_train) / 2
    imgdf = pd.DataFrame(glob.glob(img_src_dir), columns=['im'])
    #     imgdf = pd.DataFrame(glob.glob(src_dir+ '/*.png'), columns=['im'])
    #     print(imgdf.iloc[0,:])

    train, val, test = np.split(imgdf.sample(frac=1, random_state=200),
                                [int(frac_train * len(imgdf)), int(frac2 * len(imgdf))])
    train.to_csv('train.csv')
    test.to_csv('test.csv')
    val.to_csv('val.csv')
    folds = ['train', 'val', 'test']
    data = [train, val, test]

    for c, df in enumerate(data):

        img_dst_dir_ = src_dir + '/images/' + folds[c]
        # img_dst_dir_ = src_dir+'/annotations-rswcw/'+ folds[c]

        # img_dst_dir_ = src_dir+ folds[c]
        if not os.path.exists(img_dst_dir_):
            os.makedirs(img_dst_dir_)

        annot_dst_dir_ = src_dir + '/annotations/' + folds[c]
        gr_dst_dir_ = src_dir + '/annotations/' + '/gray/' + folds[c]
        # gr_dst_dir_ = src_dir+'/annotations-swcw/'+ folds[c]
        if not os.path.exists(annot_dst_dir_):
            os.makedirs(annot_dst_dir_)
        if not os.path.exists(gr_dst_dir_):
            os.makedirs(gr_dst_dir_)

        for idx, row in df.iterrows():
            shutil.copy(row.im, img_dst_dir_)
            # gr_annot = os.path.join(src_dir, 'annotations-rswcw','gray',row.im.split('/')[-1])
            # annot = os.path.join(src_dir, 'annotations-rswcw', 'gray', row.im.split('/')[-1])
            annot = os.path.join(src_dir, 'annotations', row.im.split('/')[-1])
            gr_annot = os.path.join(src_dir, 'annotations', 'gray', row.im.split('/')[-1])

            # #             print(gr_annot)
            shutil.copy(gr_annot, gr_dst_dir_)
            shutil.copy(annot, annot_dst_dir_)


# #
def createfolder(path):  # creates folder if not exists
    if path_exist(path):
        return path
    else:
        os.makedirs(path)
        return path


def find_file_startpattern(path, pattern):
    target = [f for f in os.listdir(path) if f.startswith(pattern)]
    if len(target) > 0:
        return target[0]
    else:
        return -1
    # for img in os.listdir(path):
    #     if img.startswith(pattern):
    #         return img
    #     else:
    #         continue


def generate_path(src_path, filename):
    fpath = os.path.join(src_path, filename)
    return fpath


def path_exist(path):
    if os.path.exists(path):
        return True
    else:
        return False


def read_img_folder(input_path, file_format):
    """
    reads files in the folder (excludes files starting with '.'
    :param input_path: (str)
    :param file_format: (str) png, jpg, ...
    :return: list of file names without extension
    """

    file_format = file_format.lower()
    with os.scandir(input_path) as it:
        items = [entry for entry in it if not entry.name.startswith('.')]
    # folders = [entry for entry in items if entry.is_dir()]
    files = [i for i in items if i.is_file()]
    imgs_names = [i.name.split(f'.{file_format}')[0] for i in files if i.name.endswith(file_format)]

    return imgs_names


def find_image_ends_with(image_names_ls, pattern):
    """
    take list of image names (w/o extension) and finds the file names that end in the specified pattern.
    pattern: str
"""
    found = [i for i in image_names_ls if i.endswith(pattern)]
    if len(found) == 1:
        return found[0]
    elif len(found) == 0:
        logging.error(f'No file name ends with {pattern}!')
        return -1
    else:
        logging.warning(f'Found More than one file that name ends with {pattern}!')
        return found


def disk_size_convert(size_bytes):
    """
    Convert size in bytes to proper readable units
    """
    if size_bytes == 0:
        return '0B'
    unit = ('B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB')
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    size = round(size_bytes / p, 1)
    return size, unit[i]


def get_free_space_bytes(dirname):
    """Gets directory path and return free space in that drive in Bytes."""
    usage = psutil.disk_usage(dirname)
    return usage.free


def free_space_check(dirname):
    """
    checks if there is enough space in that disk to write/download the images and raise Warnings
    e.g. NYC had around 203,000 tiles with Average size of 30 KB and total size of 5.82 GB
    the average number is low due to black and white tiles. The actual tiles were usually above 50 KB.
    So the available disk space should be taken into account before downloading and stitching

    """
    free_space_b = get_free_space_bytes(dirname)
    free_size, unit = disk_size_convert(free_space_b)
    if unit == 'MB':
        raise Warning('Low Disk Space!')
    elif unit == 'GB' and free_size < 6:
        raise Warning('Low Disk Space!')
    else:
        pass


def split_dataset(src_image_dir, src_annotation_dir, validation_ratio):
    """
    Splits a dataset of image and annotation files into train and validation sets.

    Args:
    - src_image_dir: A string specifying the directory containing the image files.
    - src_annotation_dir: A string specifying the directory containing the annotation files.
    - validation_ratio: A float specifying the ratio of files to use for validation.
    """
    # Get the parent directory of the src_image_dir directory
    parent_dir = os.path.dirname(os.path.abspath(src_image_dir))

    # Define the paths to the train and validation directories
    train_dir = os.path.join(parent_dir, 'train')
    val_dir = os.path.join(parent_dir, 'val')

    # Create the train and validation directories if they don't exist
    for new_dir in [train_dir, val_dir]:
        for subf in ['images', 'annotations']:
            temp_path = os.path.join(new_dir, subf)
            if not os.path.exists(temp_path):
                os.makedirs(temp_path)
                # print('made: ', temp_path)

    # Get a list of all the image files
    image_files = glob.glob(os.path.join(src_image_dir, "*.png"))

    # Shuffle the list of image files
    random.shuffle(image_files)

    # Split the list of image files into train and validation sets
    val_size = int(len(image_files) * validation_ratio)
    train_files = image_files[val_size:]
    val_files = image_files[:val_size]

    # Move the train files to the train directory
    copy_splitted(train_files, src_annotation_dir, train_dir)
    copy_splitted(val_files, src_annotation_dir, val_dir)

    print("Done!")


def copy_splitted(img_ls, src_annotation_dir, dest_dir):
    for img_path in img_ls:
        file_name = os.path.basename(img_path)
        src_annot_path = os.path.join(src_annotation_dir, file_name)

        dst_img_path = os.path.join(dest_dir, 'images', file_name)
        dst_annot_path = os.path.join(dest_dir, 'annotations', file_name)
        shutil.copyfile(img_path, dst_img_path)
        shutil.copyfile(src_annot_path, dst_annot_path)