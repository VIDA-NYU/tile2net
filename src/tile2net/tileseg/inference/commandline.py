import argparse

import argh

from tile2net.namespace import Namespace

arg = argparse.ArgumentParser().add_argument
globals()['arg'] = argh.arg

from toolz import compose_left

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
        '--eval_folder', type=str,
        help='The path to the folder to run inference on; if False, '
             'then it determines the eval folder from the city_info.json',
    ),
    arg(
        '--result_dir', type=str,
    ),
    arg(
        '--dump_percent',
        type=int,
        default=0,
        help='The percentage of segmentation results to save. 100 means all, 0 means none.',
    ),
    arg(
        '--assets_path', type=str,
    ),
    arg(
        '--snapshot', type=str,
        dest='model.snapshot',
    ),
    arg(
        '--hrnet_checkpoint',
        dest='model.hrnet_checkpoint',
    ),
    arg(
        '--quiet', '-q', action='store_true',
        # default=False,
        help='Suppress all output'
    ),
    arg(
        '--arch', type=str,
        # default='ocrnet.HRNet_Mscale',
        help='Network architecture'
    ),

    # for inference it is always satellite, if they want to train and create a new model with different dataset
    # it will change
    arg(
        '--dataset', type=str,
        # default='satellite',
        help='name of your dataset',
        dest='dataset.name',
    ),
    arg(
        '--crop_size', type=str,
        # default='640,640',
        help='training crop size: either scalar or h,w',
        dest='dataset.crop_size',
    ),

    # should not be set by the user in most cases
    arg(
        '--local_rank',
        # default=0,
        type=int,
        help='parameter for distributed training'
    ),
    # should not be set by the user in most cases
    arg(
        '--global_rank',
        # default=0,
        type=int,
        help='parameter used for distributed training'
    ),
    arg(
        '--world_size', type=int,
        # default=1,
    ),

    # this will always be this for our model.
    arg(
        '--trunk', type=str,
        # default='hrnetv2',
        help='trunk model, can be: hrnetv2 (default), resnet101, resnet50'
    ),

    arg(
        '--start_epoch', type=int,
        # default=0
    ),
    arg(
        '--bs_val', type=int,
        # default=1,
        help='Batch size for Validation per gpu',
        dest='model.bs_val',
    ),
    arg(
        '--restore_net', action='store_true',
        # default=False
    ),
    arg(
        '--exp', type=str,
        # default='default',
        help='experiment directory name'
    ),
    arg(
        '--syncbn', action='store_true',
        # default=False,
        help='Use Synchronized BN'
    ),
    arg(
        '--dump_augmentation_images', action='store_true',
        # default=False,
        help='Dump Augmented Images for sanity check'
    ),
    arg(
        '--test_mode', action='store_true',
        # default=False,
        help='Minimum testing to verify nothing failed, '
             'runs code for 1 epoch of train and val',
        dest='options.test_mode',
    ),
    # Multi Scale Inference
    arg(
        '--multi_scale_inference', action='store_true',
        help='Run multi scale inference',
    ),

    arg(
        '--default_scale', type=float,
        # default=1.0,
        help='default scale to run validation'
    ),
    arg(
        '--log_msinf_to_tb', action='store_true',
        # default=False,
        help='Log multi-scale Inference to Tensorboard'
    ),

    arg(
        '--eval', type=str,
        # default='test',
        help='just run evaluation, can be set to val or trn, test or folder',
        dest='model.eval',
    ),
    arg(

        '--three_scale', action='store_true',
        # default=False,
        dest='model.three_scale',
    ),

    arg(
        '--alt_two_scale', action='store_true',
        # default=False,
        dest='model.alt_two_scale',
    ),
    arg(
        '--mscale_lo_scale', type=float,
        # default=0.5,
        help='low resolution training scale',
        dest='model.mscale_lo_scale',
    ),
    arg(
        '--dump_topn', type=int,
        # default=50,
        help='Dump worst val images'
    ),

    arg(
        '--dump_assets', action='store_true',
        help='Dump interesting assets'
    ),
    arg(
        '--dump_all_images', action='store_true',
        help='Dump all images, not just a subset'
    ),

    arg(
        '--trial', type=int,
        # default=None
    ),

    arg(
        '--val_freq', type=int,
        # default=1,
        help='how often (in epochs'
             '), to run validation'),
    arg(
        '--deterministic', action='store_true',
        # default=False,

    ),

    arg(
        '--grad_ckpt', action='store_true',
        # default=False,
        dest='model.grad_ckpt',
    ),
    # train
    arg(
        '--resume', type=str,
        # default=None,
        help='continue training from a checkpoint. weights, '
             'optimizer, schedule are restored',
    ),
    # train
    arg(
        '--max_epoch', type=int,
        # default=300,
    ),
    # train
    arg(
        '--max_cu_epoch', type=int,
        # default=150,
        help='Class Uniform Max Epochs'
    ),
    # train
    arg(
        '--rand_augment',
        # default=None,
        help='RandAugment setting: set to \'N,M\'',
        dest='model.rand_augment',
    ),
    # train
    arg(
        '--init_decoder',
        # default=False,
        action='store_true',
        help='initialize decoder with kaiming normal',
        dest='options.init_decoder',
    ),
    # train
    arg(
        '--scale_min', type=float,
        # default=0.5,
        help='dynamically scale training images down to this size',
        dest='model.scale_min',
    ),
    # train
    arg(
        '--scale_max', type=float,
        # default=2.0,
        help='dynamically scale training images up to this size',
        dest='model.scale_max',
    ),
    # train
    arg(
        '--rescale', type=float,
        # default=1.0,
        help='Warm Restarts new lr ratio compared to original lr'
    ),
    # train
    arg(
        '--supervised_mscale_loss_wt', type=float,
        help='weighting for the supervised loss',
        dest='loss.supervised_mscale_wt',
    ),
    # train
    arg(
        '--ocr_aux_loss_rmi', action='store_true',
        # default=False,
        help='allow rmi for aux loss'
    ),
    # train
    arg(
        '--tau_factor', type=float,
        # default=1,
        help='Factor for NASA optimization function',

    ),
    # train
    arg(
        '--extra_scales', type=str,
        # default='0.5,1.5,2.0',
        dest='model.extra_scales',
    ),
    # train
    arg(
        '--n_scales', type=str,
        # default=None,
        dest='model.n_scales',
    ),
    # train
    arg(
        '--translate_aug_fix', action='store_true',
        # default=False,
        dest='dataset.translate_aug_fix',
    ),
    # train

    # tile2net
    arg(
        '--tile2net',
        # default=True,
        action='store_false', dest='tile2net',
        help='if true, creates the polygons and lines from the results',

    ),
    # tile2net
    arg(
        '--boundary_path', type=str,
        # default=None,
        help='path to the boundary file shapefile'
    ),
    arg(
        '--interactive',
        action='store_true',
        help='tile2net is being run in interactive python'
    ),
    arg(
        '--debug', action='store_true'
    ),
    arg(
        '--local', action='store_true'
    ),
    arg(
        '--remote', action='store_true'
    ),
)


