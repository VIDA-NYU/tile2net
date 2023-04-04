import os
from tile2net.raster.tile_utils.genutils import path_exist, free_space_check
import json

def proj_name_handler(proj_name):
    msg_list = ['Avoid These in project name: ']
    if ' ' in proj_name:
        msg_list.append(' " " (space) character! ')

    if '.' in proj_name:
        msg_list.append(' "." character! ')
    if len(proj_name) > 15:
        msg_list.append('using long names for project since it will '
                        'be added to beginning of each image name!')

    if len(msg_list) > 1:
        msg = ' '.join(msg_list)
        raise ValueError({msg})
    else:
        pass

def initial_path_handler(proj_path):
    if path_exist(proj_path):
        free_space_check(proj_path)

    else:
        os.makedirs(proj_path)
        free_space_check(proj_path)


def dir_construct(p_dict: dict, base_key: str, dirs: list, suff: str = None):
    base_path = p_dict[base_key]
    for d in dirs:
        new_path = os.path.join(base_path, d)
        if os.path.exists(new_path):
            if len(os.listdir(new_path)) > 0:
                print(f'directory {new_path} already exist!')
            else:
                pass
        else:
            os.makedirs(new_path)

        if suff is not None:
            key = f'{d}{suff}'
        else:
            key = d

        p_dict.update({key: new_path})

def create_base_dir_structure(p_dict: dict):
    dirs = ['tiles', 'polygons', 'network', 'segmentation']
    dir_construct(p_dict, 'proj_path', dirs, suff='_main')

def proj_path_handler(proj_name, proj_path, repo_path=None):
    """
    sets the folder path based on the user defined project name and tile size and zoom level
    and check if it exists
    """
    # initial checks
    proj_name_handler(proj_name)
    initial_path_handler(proj_path)

    p_dict = dict()

    if os.path.basename(proj_path) == proj_name:
        main_dir_path = proj_path
    else:
        main_dir_path = os.path.join(proj_path, proj_name)

    if os.path.exists(main_dir_path):
        pass
    else:
        os.makedirs(main_dir_path)

    p_dict['proj_name'] = proj_name
    p_dict['proj_path'] = main_dir_path
    create_base_dir_structure(p_dict)
    create_segment_dir_structure(p_dict)
    if repo_path:
        p_dict['config'] = os.path.join(repo_path, 'tileseg', 'config.py')
    with open(os.path.join(main_dir_path, f'{proj_name}_project_structure.json'), 'w+') as f:
        json.dump(p_dict, f)
        f.close()
    return p_dict

def create_segment_dir_structure(p_dict):
    dirs = ['model_assets', 'segm_result']
    dir_construct(p_dict, 'segmentation_main', dirs, suff=None)

    seg_dir = ['seg_weights']
    dir_construct(p_dict, 'model_assets', seg_dir, suff=None)


def modify_config(p_dict):
    config_path = p_dict['config']
    with open(config_path, 'r') as a:
        text = a.readlines()
    asset = []
    result = []
    city = []
    for c, i in enumerate(text):
        if '__C.ASSETS_PATH =' in i:
            asset.append((i, c))
        elif '__C.RESULT_DIR =' in i:
            result.append((i, c))
        elif '__C.CITY_INFO_PATH =' in i:
            city.append((i, c))

    if len(asset) == 1:
        line_asset = asset[0][0].split('=')[0]
        line_asset_idx = asset[0][1]
        new_line_asset = f"{line_asset}= r'{p_dict['model_assets']}'\n"
        text[line_asset_idx] = new_line_asset
        with open(config_path, 'w+') as f:
            f.writelines(text)
        a_success = True
    else:
        a_success = False
        raise ValueError('Corrupted Config.py File!')

    if len(result) == 2:
        line_res = result[0][0].split('=')[0]
        line_res_idx = result[0][1]
        new_line_res = f"{line_res}= r'{p_dict['segm_result']}'\n"
        text[line_res_idx] = new_line_res
        with open(config_path, 'w+') as f:
            f.writelines(text)
        r_success = True
    else:
        print(result)
        r_success = False
        raise ValueError('Corrupted Config.py File!')

    if len(city) == 1:
        line_res = city[0][0].split('=')[0]
        line_res_idx = city[0][1]
        new_line_res = f"{line_res}= r'{p_dict['city_info_path_path']}'\n"
        text[line_res_idx] = new_line_res
        with open(config_path, 'w+') as f:
            f.writelines(text)
        r_success = True
    else:
        r_success = False
        print(city)
        raise ValueError('Corrupted Config.py File!')

    if a_success and  r_success:
        print('"Config.Py" modified Successfully!')