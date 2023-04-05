import logging
import os
import glob
import math
import shutil
from PIL import Image
import psutil
import random


def deg2num(lat_deg, lon_deg, zoom):
    """
    converts lat/lon to pixel coordinates in given zoom of the EPSG:3857 pyramid
    Parameters
    ----------
    lat_deg: float
        latitude in degrees
    lon_deg: float
        longitude in degrees
    zoom: int
        zoom level of the tile

    Returns
    -------
    xtile: int
        xcoodrinate of the tile in xyz system
    ytile: int
        ycoodrinate of the tile in xyz system
    """
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile: float = int((lon_deg + 180.0) / 360.0 * n)
    ytile: float = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def num2deg(xtile, ytile, zoom):
    """
    converts pixel coordinates in given zoom of the EPSG:3857 pyramid to lat/lon
    Parameters
    ----------
    xtile: int
        xcoodrinate of the tile in xyz system
    ytile: int
        ycoodrinate of the tile in xyz system
    zoom: int
        zoom level of the tile

    Returns
    -------
    lat_deg: float

    lon_deg: float


    """
    n = 2.0 ** zoom
    lon_deg: float = xtile / n * 360.0 - 180.0
    lat_rad: float = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg: float = math.degrees(lat_rad)
    return lat_deg, lon_deg


def path_check(pathchk):
    """
    Checks if the output file exists and asks the user if he wants to overwrite it or not
    Parameters
    ----------
    pathchk

    Returns
    -------
    pathchk : str
        The path to the output file
    """
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
    """
    Changes the name of the images in the folder
    Parameters
    ----------
    src_path: str
        path to the folder
    args: str
        name of the class
    """
    imgs = glob.glob(src_path)
    dest_path = src_path[:-6] + '/new_name_class/'
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    for c, im in enumerate(imgs):
        img = Image.open(im)
        img.save(dest_path + str(c) + '-' + '.png')


def move_img_ann(img_src, ann_src, ann_dest):
    """
    Moves the images and annotations to a new folder
    Parameters
    ----------
    img_src : str
        path to the image folder
    ann_src: str
        path to the annotation folder
    ann_dest: str
        path to the destination folder
    """
    # ann_dest = os.path.join()
    if not os.path.exists(ann_dest):
        os.makedirs(ann_dest)

    for i in os.listdir(img_src):
        if i in os.listdir(ann_src):
            shutil.copy(os.path.join(ann_src, i), ann_dest)


def move_ann(src, ann_src):
    """
    Moves the images and annotations to a new folder
    Parameters
    ----------
    src : str
        path to the folder
    ann_src: str
        path to the annotation folder
    """
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



# #
def createfolder(path):  # creates folder if not exists
    """
    Creates a folder if it does not exist
    Parameters
    ----------
    path : str
        path to the folder

    Returns
    -------
    path : str
        path to the folder
    """
    if path_exist(path):
        return path
    else:
        os.makedirs(path)
        return path


def find_file_startpattern(path, pattern):
    """
    Finds the file in the folder that starts with the pattern
    Parameters
    ----------
    path : str
        path to the folder
    pattern : str
        pattern to search for

    Returns
    -------
    target : str
        name of the file
    """
    target = [f for f in os.listdir(path) if f.startswith(pattern)]
    if len(target) > 0:
        return target[0]
    else:
        return -1


def generate_path(src_path, filename):
    """
    Generates the path of the file
    Parameters
    ----------
    src_path : str
        path to the folder
    filename : str
        name of the file
    Returns
    -------
    fpath : str
        path to the file
    """
    fpath = os.path.join(src_path, filename)
    return fpath


def path_exist(path):
    """
    Checks if the path exists
    Parameters
    ----------
    path : str
        path to the folder

    Returns
    -------
    bool
        True if the path exists
    """
    if os.path.exists(path):
        return True
    else:
        return False


def read_img_folder(input_path, file_format):
    """
    Reads the images in the folder
    Parameters
    ----------
    input_path  : str
        path to the folder
    file_format : str
        format of the image

    Returns
    -------
    imgs_names : list
        list of image names
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
    Finds the image that ends with the pattern
    Parameters
    ----------
    image_names_ls : list
        list of image names
    pattern

    Returns
    -------

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
    Converts the size of the disk to human readable format
    Parameters
    ----------
    size_bytes  : int
        size of the disk in bytes

    Returns
    -------
    size
    unit

    """
    if size_bytes == 0:
        return '0B'
    unit = ('B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB')
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    size = round(size_bytes / p, 1)
    return size, unit[i]


def get_free_space_bytes(dirname):
    """

    Parameters
    ----------
    dirname : str or path-like object
        path to the directory

    Returns
    -------
    free space in bytes
    """
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
    """
    Copies the splited images and annotations to the destination directory
    Parameters
    ----------
    img_ls: list
        list of image paths
    src_annotation_dir: str
        path to the annotation directory
    dest_dir: str
        path to the destination directory

    Returns
    -------
    None

    """
    for img_path in img_ls:
        file_name = os.path.basename(img_path)
        src_annot_path = os.path.join(src_annotation_dir, file_name)

        dst_img_path = os.path.join(dest_dir, 'images', file_name)
        dst_annot_path = os.path.join(dest_dir, 'annotations', file_name)
        shutil.copyfile(img_path, dst_img_path)
        shutil.copyfile(src_annot_path, dst_annot_path)
