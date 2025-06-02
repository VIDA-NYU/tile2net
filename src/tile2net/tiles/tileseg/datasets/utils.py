import os
import numpy as np
from PIL import Image
from tile2net.logger import logger


def make_dataset_folder(folder, testing=None):
    """
    Create Filename list for images in the provided path

    input: path to directory with *only* images files
	   test_mode: for test only with no ground truth 
   
    returns: items list with None filled for mask path
    """
    items = os.listdir(folder)
    if testing:
                
        items = [(os.path.join(folder, f), '') for f in items]
    else:
        mask_root = folder.replace('images', 'annotations', 6)
        items = [(os.path.join(folder, f), os.path.join(mask_root, f)) for f in items]
    
    items = sorted(items)
    # print(f'Found {len(items)} folder imgs')
    logger.debug(f'Found {len(items)} folder imgs')

    """
    orig_len = len(items)
    rem = orig_len % 8
    if rem != 0:
        items = items[:-rem]

    msg = 'Found {} folder imgs but altered to {} to be modulo-8'
    msg = msg.format(orig_len, len(items))
    print(msg)
    """

    return items
