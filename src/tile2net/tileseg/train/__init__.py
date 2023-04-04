"""
Copyright 2020 Nvidia Corporation
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import absolute_import
from __future__ import division

import argparse
import os
import sys
import time
import torch
from math import sqrt
from torch.cuda import amp
from runx.logx import logx
from tile2net.tileseg.config import assert_and_infer_cfg, update_epoch, cfg
from tile2net.tileseg.utils.misc import AverageMeter, prep_experiment, eval_metrics
from tile2net.tileseg.utils.misc import ImageDumper
from tile2net.tileseg.utils.trnval_utils import eval_minibatch, validate_topn
from tile2net.tileseg.loss.utils import get_loss
from tile2net.tileseg.loss.optimizer import get_optimizer, restore_opt, restore_net
from tile2net.tileseg import datasets
from tile2net.tileseg import network

from tile2net.tileseg.train.commandline import commandline
from tile2net.namespace import Namespace

import pandas as pd
import numpy as np
import json
import copy
import datetime

# from tile2net.Data import Data
from tile2net.raster.pednet import PedNet


# Import autoresume module
sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))
AutoResume = None
# try:
#     from userlib.auto_resume import AutoResume
# except ImportError:
#     print(AutoResume)

@commandline
def train(args: Namespace):
    args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                        'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}

    # Enable CUDNN Benchmarking optimization
    torch.backends.cudnn.benchmark = True
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    args.world_size = 1

    # Test Mode run two epochs with a few iterations of training and val
    if args.options.test_mode:
        args.max_epoch = 2

    if 'WORLD_SIZE' in os.environ and args.model.apex:
        # args.model.apex = int(os.environ['WORLD_SIZE']) > 1
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.global_rank = int(os.environ['RANK'])

    if args.model.apex:
        print('Global Rank: {} Local Rank: {}'.format(
            args.global_rank, args.local_rank))
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
            init_method='env://')


    def check_termination(epoch):
        if AutoResume:
            shouldterminate = AutoResume.termination_requested()
            if shouldterminate:
                if args.global_rank == 0:
                    progress = "Progress %d%% (epoch %d of %d)" % (
                        (epoch * 100 / args.max_epoch),
                        epoch,
                        args.max_epoch
                    )
                    AutoResume.request_resume(
                        user_dict={"RESUME_FILE": logx.save_ckpt_fn,
                                   "TENSORBOARD_DIR": args.result_dir,
                                   "EPOCH": str(epoch)
                                   }, message=progress)
                    return 1
                else:
                    return 1
        return 0


    def main():
        """
        Main Function
        """
        if AutoResume:
            AutoResume.init()

        assert args.result_dir is not None, 'need to define result_dir arg'
        logx.initialize(logdir=args.result_dir,
            tensorboard=True, hparams=vars(args),
            global_rank=args.global_rank)

        # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
        assert_and_infer_cfg(args)
        prep_experiment(args)
        train_loader, val_loader, train_obj = \
            datasets.setup_loaders(args)
        criterion, criterion_val = get_loss(args)

        auto_resume_details = None
        if AutoResume:
            auto_resume_details = AutoResume.get_resume_details()

        if auto_resume_details:
            checkpoint_fn = auto_resume_details.get("RESUME_FILE", None)
            checkpoint = torch.load(checkpoint_fn,
                map_location=torch.device('cpu'))
            args.result_dir = auto_resume_details.get("TENSORBOARD_DIR", None)
            args.start_epoch = int(auto_resume_details.get("EPOCH", None)) + 1
            args.restore_net = True
            args.restore_optimizer = True
            msg = ("Found details of a requested auto-resume: checkpoint={}"
                   " tensorboard={} at epoch {}")
            logx.msg(msg.format(checkpoint_fn, args.result_dir,
                args.start_epoch))
        elif args.resume:
            checkpoint = torch.load(args.resume,
                map_location=torch.device('cpu'))
            args.arch = checkpoint['arch']
            args.start_epoch = int(checkpoint['epoch']) + 1
            args.restore_net = True
            args.restore_optimizer = True
            msg = "Resuming from: checkpoint={}, epoch {}, arch {}"
            logx.msg(msg.format(args.resume, args.start_epoch, args.arch))
        elif args.model.snapshot:
            if 'ASSETS_PATH' in args.model.snapshot:
                args.model.snapshot = args.model.snapshot.replace('ASSETS_PATH', cfg.ASSETS_PATH)
            checkpoint = torch.load(args.model.snapshot,
                map_location=torch.device('cpu'))
            args.restore_net = True
            msg = "Loading weights from: checkpoint={}".format(args.model.snapshot)
            logx.msg(msg)

        net = network.get_net(args, criterion)
        optim, scheduler = get_optimizer(args, net)
        # define the NASA optimizer parameter
        # iter_tot = len(train_loader)*args.max_epoch
        #    tau = args.tau_factor/sqrt(iter_tot)
        tau = 1
        # k = 1
        # optim, scheduler = get_optimizer(args, net, tau, k)
        if args.train.fp16:
            net, optim = amp.initialize(net, optim, opt_level=args.amp_opt_level)

        net = network.wrap_network_in_dataparallel(net, args.model.apex)

        if args.restore_optimizer:
            restore_opt(optim, checkpoint)
        if args.restore_net:
            restore_net(net, checkpoint)

        if args.options.init_decoder:
            net.module.init_mods()

        torch.cuda.empty_cache()

        if args.start_epoch != 0:
            scheduler.step(args.start_epoch)

        if args.city_info_path:
            boundary_path = args.boudary_path

            with open(args.city_info_path) as F:
                g = json.load(F)
            city_data = PedNet(tuple(g['bbox']), g['name'], g['num_class'], size=g['size'],
                zoom=g['zoom'], crs=g['crs'],
                stitch_step=g['stitch_step'], boundary_path=boundary_path, filter=None,
                padding=None)
        # There are 4 options for evaluation:
        #  --eval val                           just run validation
        #  --eval val --dump_assets             dump all images and assets
        #  --eval folder                        just dump all basic images
        #  --eval folder --dump_assets          dump all images and assets

        if args.model.eval== 'test':
            validate(val_loader, net, criterion=None, optim=None, epoch=0,
                calc_metrics=False, dump_assets=args.dump_assets,
                dump_all_images=True, testing=True, grid=city_data)
            return 0

        if args.model.eval== 'val':

            if args.dump_topn:
                validate_topn(val_loader, net, criterion_val, optim, 0, args)
            else:
                validate(val_loader, net, criterion=criterion_val, optim=optim, epoch=0,
                    dump_assets=args.dump_assets,
                    dump_all_images=args.dump_all_images,
                    calc_metrics=args.calc_metrics)
            return 0

        elif args.model.eval== 'folder':
            # Using a folder for evaluation means to not calculate metrics
            validate(val_loader, net, criterion=criterion_val, optim=optim, epoch=0,
                calc_metrics=False, dump_assets=args.dump_assets,
                dump_all_images=True)
            return 0

        elif args.model.eval is not None:
            raise 'unknown eval option {}'.format(args.eval)

        for epoch in range(args.start_epoch, args.max_epoch):
            update_epoch(epoch)

            if args.only_coarse:
                train_obj.only_coarse()
                train_obj.build_epoch()
                if args.model.apex:
                    train_loader.sampler.set_num_samples()

            elif args.class_uniform_pct:
                if epoch >= args.max_cu_epoch:
                    train_obj.disable_coarse()
                    train_obj.build_epoch()
                    if args.model.apex:
                        train_loader.sampler.set_num_samples()
                else:
                    train_obj.build_epoch()
            else:
                pass

            train(train_loader, net, optim, epoch)

            if args.model.apex:
                train_loader.sampler.set_epoch(epoch + 1)

            if epoch % args.val_freq == 0:
                validate(val_loader, net, criterion_val, optim, epoch)

            scheduler.step()

            if check_termination(epoch):
                return 0


    def train(train_loader, net, optim, curr_epoch):
        """
        Runs the training loop per epoch
        train_loader: Data loader for train
        net: thet network
        optimizer: optimizer
        curr_epoch: current epoch
        return:
        """
        net.train()

        train_main_loss = AverageMeter()
        start_time = None
        warmup_iter = 10
        loss_metric = dict([('epoch', []), ('loss', []), ('lr', [])])
        for i, data in enumerate(train_loader):
            if i <= warmup_iter:
                start_time = time.time()
            # inputs = (bs,3,713,713)
            # gts    = (bs,713,713)
            images, gts, _img_name, scale_float = data
            batch_pixel_size = images.size(0) * images.size(2) * images.size(3)
            images, gts, scale_float = images.cuda(), gts.cuda(), scale_float.cuda()
            inputs = {'images': images, 'gts': gts}

            optim.zero_grad()
            main_loss = net(inputs)

            if args.model.apex:
                log_main_loss = main_loss.clone().detach_()
                torch.distributed.all_reduce(log_main_loss,
                    torch.distributed.ReduceOp.SUM)
                log_main_loss = log_main_loss / args.world_size
            else:
                main_loss = main_loss.mean()
                log_main_loss = main_loss.clone().detach_()

            train_main_loss.update(log_main_loss.item(), batch_pixel_size)
            if args.train.fp16:
                with amp.scale_loss(main_loss, optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                main_loss.backward()

            optim.step()

            if i >= warmup_iter:
                curr_time = time.time()
                batches = i - warmup_iter + 1
                batchtime = (curr_time - start_time) / batches
            else:
                batchtime = 0

            msg = ('[epoch {}], [iter {} / {}], [train main loss {:0.6f}],'
                   ' [lr {:0.6f}] [batchtime {:0.3g}]')
            msg = msg.format(
                curr_epoch, i + 1, len(train_loader), train_main_loss.avg,
                optim.param_groups[-1]['lr'], batchtime)
            logx.msg(msg)

            metrics = {'loss': train_main_loss.avg,
                       'lr': optim.param_groups[-1]['lr']}
            curr_iter = curr_epoch * len(train_loader) + i
            logx.metric('train', metrics, curr_iter)
            loss_metric['epoch'].append(curr_epoch)
            loss_metric['loss'].append(train_main_loss.avg)
            loss_metric['lr'].append(optim.param_groups[-1]['lr'])

            if i >= 10 and args.options.test_mode:
                del data, inputs, gts
                return
            del data

    def validate(val_loader, net, criterion, optim, epoch,
        calc_metrics=True,
        dump_assets=False,
        dump_all_images=False, testing=None, grid=None):
        """
        Run validation for one epoch
        :val_loader: data loader for validation
        :net: the network
        :criterion: loss fn
        :optimizer: optimizer
        :epoch: current epoch
        :calc_metrics: calculate validation score
        :dump_assets: dump attention prediction(s) images
        :dump_all_images: dump all images, not just N
        """
        dumper = ImageDumper(val_len=len(val_loader),
            dump_all_images=dump_all_images,
            dump_assets=dump_assets,
            dump_for_auto_labelling=args.dump_for_auto_labelling)

        net.eval()
        val_loss = AverageMeter()
        iou_acc = 0
        pred = dict()
        _temp = dict.fromkeys([i for i in range(10)], None)
        for val_idx, data in enumerate(val_loader):
            input_images, labels, img_names, _ = data
            if args.dump_for_auto_labelling:
                submit_fn = '{}.png'.format(img_names[0])
                if val_idx % 20 == 0:
                    logx.msg(f'validating[Iter: {val_idx + 1} / {len(val_loader)}]')
                if os.path.exists(os.path.join(dumper.save_dir, submit_fn)):
                    continue

            # Run network
            assets, _iou_acc = \
                eval_minibatch(data, net, criterion, val_loss, calc_metrics,
                    args, val_idx)

            iou_acc += _iou_acc

            input_images, labels, img_names, _ = data

            if testing:
                prediction = assets['predictions'][0]
                values, counts = np.unique(prediction, return_counts=True)
                pred[img_names[0]] = copy.copy(_temp)
                for v in range(len(values)):
                    pred[img_names[0]][values[v]] = counts[v]

                dumper.dump(
                    {'gt_images': labels, 'input_images': input_images, 'img_names': img_names,
                     'assets': assets},
                    val_idx, testing=True, grid=grid)
            else:
                dumper.dump({'gt_images': labels,
                             'input_images': input_images,
                             'img_names': img_names,
                             'assets': assets}, val_idx)

            if val_idx > 5 and args.options.test_mode:
                break

            if val_idx % 20 == 0:
                logx.msg(f'validating[Iter: {val_idx + 1} / {len(val_loader)}]')

        was_best = False
        if calc_metrics:
            was_best = eval_metrics(iou_acc, args, net, optim, val_loss, epoch)

        # Write out a summary html page and tensorboard image table
        if not args.dump_for_auto_labelling:
            dumper.write_summaries(was_best)

        if testing:
            sufix = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")
            df = pd.DataFrame.from_dict(pred, orient='index')
            df.to_csv(os.path.join(dumper.save_dir, f'raw_numb_{sufix}.csv'), mode='a+')
            df_p = df.div(df.sum(axis=1), axis=0)
            df_p.to_csv(os.path.join(dumper.save_dir, f'freq_{sufix}.csv'), mode='a+')

            grid.save_ntw_polygon(dumper.save_dir)

    #        grid.save_ntw_line(dumper.save_dir)
    #        grid.convert_whole_poly2line(dumper.save_dir)
    #        grid.post_process(grid.ntw_line, dumper.save_dir, 4, 8)

    main()
