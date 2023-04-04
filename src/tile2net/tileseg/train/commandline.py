import argparse

from typing import Optional

import argh
from toolz import compose_left


arg = argparse.ArgumentParser().add_argument
globals()['arg'] = argh.arg

parser = argparse.ArgumentParser(description='Semantic Segmentation')

from tile2net.raster.project import Project
from tile2net.namespace import Namespace


commandline = compose_left(
    argh.expects_obj,
    Namespace.wrap,
    arg(
        '--city_info', type=str,
        help='The path to the city_info.json; if False, then it parses the JSON '
             'from the stdout',
        dest='city_info_path',
    ),
    arg(
        '--result_dir',
        help='Where output results are written',
    ),
    arg(
        '--eval_folder', type=str,
        help='The path to the eval folder; if None, then it determines the eval folder '
             'from the city_info.json',
        dest='eval_folder'
    ),
    arg(
        '--assets_path',
    ),
    arg(
        '--snapshot',
        dest='model.snapshot',
    ),
    arg(
        '--hrnet_checkpoint',
        dest='model.hrnet_checkpoint',
    ),
    arg(
        '--quiet', '-q', action='store_true', help='Suppress all output'
    ),
    arg(
        '--lr', type=float, )
    ,
    arg(
        '--arch', type=str, help='Network architecture'
    ),
    arg(
        '--old_data', action='store_true', help='sets the dataset to the first one in hrnet'
    ),
    arg(
        '--dataset', type=str, help='name of your dataset',
        dest='dataset.name',
    ),
    arg(
        '--num_workers', type=int, help='cpu worker threads per dataloader instance'
    ),
    arg(
        '--cv', type=int, help='Cross-validation split id to use. Default # of splits set'
                               ' to 3 in config',
    ),

    arg(
        '--class_uniform_pct', type=float,
        help='What fraction of images is uniformly sampled'
    ),
    arg(
        '--class_uniform_tile', type=int,
        help='tile size for class uniform sampling'
    ),
    arg(
        '--coarse_boost_classes', type=str,
        help='Use coarse annotations for specific classes'
    ),

    arg(
        '--custom_coarse_dropout_classes', type=str,
        help='Drop some classes from auto-labelling'
    ),

    arg(
        '--img_wt_loss', action='store_true',
        help='per-image class-weighted loss'
    ),
    arg(
        '--rmi_loss', action='store_true',
        help='use RMI loss'
    ),
    arg(
        '--batch_weighting', action='store_true',
        help='Batch weighting for class (use nll class weighting using '
             'batch stats'
    ),

    arg(
        '--rescale', type=float,
        help='Warm Restarts new lr ratio compared to original lr'
    ),
    arg(
        '--repoly', type=float,
        help='Warm Restart new poly exp'
    ),

    arg(
        '--apex', action='store_true',
        help='Use Nvidia Apex Distributed Data Parallel'
    ),
    arg(
        '--fp16', action='store_true',
        help='Use Nvidia Apex AMP'
    ),

    arg(
        '--local_rank', type=int,
        help='parameter used by apex library'
    ),
    arg(
        '--global_rank', type=int,
        help='parameter used by apex library'
    ),

    arg(
        '--optimizer', type=str, help='optimizer'
    ),
    arg(
        '--amsgrad', action='store_true', help='amsgrad for adam'
    ),

    arg(
        '--freeze_trunk', action='store_true',
    ),
    arg(
        '--hardnm', type=int,
        help='0 means no aug, 1 means hard negative mining '
             'iter 1, 2 means hard negative mining iter 2'
    ),
    arg(
        '--trunk', type=str,
        help='trunk model, can be: resnet101 ('
             'default),, resnet50'
    ),
    arg(
        '--max_epoch', type=int, ),
    arg(
        '--max_cu_epoch', type=int,
        help='Class Uniform Max Epochs'
    ),
    arg(
        '--start_epoch', type=int,
    ),
    arg(
        '--color_aug', type=float,
        help='level of color augmentation'
    ),
    arg(
        '--gblur', action='store_true',
        help='Use Guassian Blur Augmentation'
    ),
    arg(
        '--bblur', action='store_true',
        help='Use Bilateral Blur Augmentation'
    ),
    arg(
        '--brt_aug', action='store_true',
        help='Use brightness augmentation'
    ),
    arg(
        '--lr_scheduler', type=str,
        help='name of lr schedule: poly',
        dest='model.lr_scheduler'
    ),
    arg(
        '--poly_exp', type=float,
        help='polynomial LR exponent'
    ),
    arg(
        '--poly_step', type=int,
        help='polynomial epoch step'
    ),
    arg(
        '--bs_trn', type=int,
        help='Batch size for training per gpu'
    ),
    arg(
        '--bs_val', type=int,
        help='Batch size for Validation per gpu'
    ),
    arg(
        '--crop_size', type=str, help='training crop size: either scalar or h,w',
        dest='dataset.crop_size'
    ),

    arg(
        '--scale_min', type=float,
        help='dynamically scale training images down to this size'
    ),
    arg(
        '--scale_max', type=float,
        help='dynamically scale training images up to this size'
    ),

    arg(
        '--weight_decay', type=float,
    ),
    arg(
        '--momentum', type=float,
    ),
    arg(
        '--resume', type=str,
        help='continue training from a checkpoint. weights, '
             'optimizer, schedule are restored'
    ),
    arg(
        '--restore_optimizer', action='store_true',
    ),
    arg(
        '--restore_net', action='store_true',
    ),
    arg(
        '--exp', type=str,
        help='experiment directory name'
    ),
    arg(
        '--syncbn', action='store_true',
        help='Use Synchronized BN'
    ),
    arg(
        '--dump_augmentation_images', action='store_true',
        help='Dump Augmentated Images for sanity check'
    ),
    arg(
        '--test_mode', action='store_true',
        help='Minimum testing to verify nothing failed, '
             'Runs code for 1 epoch of train and val',
        dest='options.test_mode'
    ),
    arg(
        '-wb', '--wt_bound', type=float,
        help='Weight Scaling for the losses'
    ),

    arg(
        '--scf', action='store_true',
        help='scale correction factor'
    ),
    # Full Crop Training
    arg(
        '--full_crop_training', action='store_true',
        help='Full Crop Training'
    ),

    # Multi Scale Inference
    arg('--multi_scale_inference', action='store_true',
        help='Run multi scale inference'
    ),

    arg(
        '--default_scale', type=float,
        help='default scale to run validation'
    ),

    arg(
        '--log_msinf_to_tb', action='store_true',
        help='Log multi-scale Inference to Tensorboard'
    ),

    arg(
        '--eval', type=str,
        help='just run evaluation, can be set to val or trn or '
             'folder'
    ),
    arg(
        '--three_scale', action='store_true',
        dest='model.three_scale',
    ),

    arg(
        '--alt_two_scale', action='store_true',
        dest='model.alt_two_scale',
    ),
    arg(
        '--do_flip', action='store_true', ),
    arg(
        '--extra_scales', type=str,
        dest='model.extra_scales',
    ),
    arg(
        '--n_scales', type=str,
        dest='model.n_scales',
    ),
    arg(
        '--align_corners', action='store_true',
    ),
    arg(
        '--translate_aug_fix', action='store_true',
        dest='dataset.translate_aug_fix',
    ),
    arg(
        '--mscale_lo_scale', type=float,
        dest='model.mscale_lo_scale',
        help='low resolution training scale'),
    arg(
        '--pre_size', type=int,
        help='resize long edge of images to this before'
             ' augmentation'
    ),
    arg(
        '--amp_opt_level', type=str,
        help='amp optimization level'
    ),
    arg(
        '--rand_augment',
        help='RandAugment setting: set to \'N,M\''
    ),
    arg(
        '--init_decoder', action='store_true',
        dest='options.init_decoder',
        help='initialize decoder with kaiming normal'
    ),
    arg(
        '--dump_topn', type=int,
        help='Dump worst val images'),
    arg(
        '--dump_assets', action='store_true',
        help='Dump interesting assets'
    ),
    arg(
        '--dump_all_images', action='store_true',
        help='Dump all images, not just a subset'
    ),

    arg(
        '--dump_for_auto_labelling', action='store_true',
        help='Dump assets for autolabelling'
    ),
    arg(
        '--dump_topn_all', action='store_true',
        help='dump topN worst failures'
    ),
    arg(
        '--custom_coarse_prob', type=float,
        help='Custom Coarse Prob'
    ),
    arg(
        '--only_coarse', action='store_true',
    ),

    arg(
        '--ocr_aspp', action='store_true',
    ),
    arg(
        '--map_crop_val', action='store_true',
    ),
    arg(
        '--aspp_bot_ch', type=int,
    ),
    arg(
        '--trial', type=int,
    ),
    arg(
        '--mscale_cat_scale_flt', action='store_true',

    ),
    arg(
        '--mscale_dropout', action='store_true',

    ),
    arg(
        '--mscale_no3x3', action='store_true',
        help='no inner 3x3'
    ),
    arg(
        '--mscale_old_arch', action='store_true',
        help='use old attention head'
    ),
    arg(
        '--mscale_init', type=float,
        help='default attention initialization'
    ),
    arg(
        '--attnscale_bn_head', action='store_true',

    ),
    arg(
        '--ocr_alpha', type=float,
        help='set HRNet OCR auxiliary loss weight'
    ),
    arg(
        '--val_freq', type=int,
        help='how often ('
             'in epochs), to run validation'
    ),
    arg(
        '--deterministic', action='store_true',

    ),
    arg(
        '--summary', action='store_true',

    ),
    arg(
        '--segattn_bot_ch', type=int,
        help='bottleneck channels for seg and attn heads'
    ),
    arg(
        '--grad_ckpt', action='store_true',
        dest='model.grad_ckpt',

    ),
    arg(
        # '--no_metrics', action='store_true',
        # help='prevent calculation of metrics'
        '--no_metrics', action='store_false', dest='calc_metrics',
    ),
    arg(
        '--supervised_mscale_loss_wt', type=float,
        dest='loss.supervised_mscale_loss_wt',
        help='weighting for the supervised loss'
    ),
    arg(
        '--ocr_aux_loss_rmi', action='store_true',
        help='allow rmi for aux loss'
    ),
    arg(
        '--tau_factor', type=float,
        help='Factor for NASA optimization function'
    ),
    arg(
        '--city_info_path', type=str,
        help='path to the json file with grid information to create an instance of class Grid'
    ),
    arg(
        '--boundary_path', type=str,
        help='path to the boundary file shapefile'
    ),
)
