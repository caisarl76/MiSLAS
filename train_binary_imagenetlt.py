import argparse
import os
import random
import shutil
import sys
import numpy as np
import warnings
import pprint

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms

from datasets.cifar10 import CIFAR10_LT
from datasets.cifar100 import CIFAR100_LT
from datasets.imagenet import ImageNet_LT

from utils import create_logger
from utils import AverageMeter, ProgressMeter
from utils import accuracy, calibration

from methods import mixup_data, mixup_criterion
from binary.reactnet_imagenet import reactnet
from binary import KD_loss
from binary.utils import *

def parse_args():
    parser = argparse.ArgumentParser("birealnet18")
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--epochs', type=int, default=256, help='num of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
    parser.add_argument('--data', metavar='DIR', help='path to dataset')
    parser.add_argument('--dataset', default='imagenet_lt', help='choose dataset to use')
    parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--pretrained', type=str, default='resnet34', help='path of ImageNet')
    parser.add_argument('-j', '--workers', default=40, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--world_size', type=int, default=-1)
    parser.add_argument('--rank', type=int, default=-1)
    parser.add_argument('--dist_url', type=str, default='tcp://224.66.41.62:23456')
    parser.add_argument('--dist_backend', type=str, default='nccl')
    parser.add_argument('--multiprocessing_distributed', action='store_true')

    args = parser.parse_args()

    return args

best_acc1 = 0
its_ece = 100
CLASSES = 1000

def main():
    args = parse_args()
    logger, writer = create_logger(args)
    logger.info('\n' + pprint.pformat(args))
    logger.info('\n' + str(args))

    if config.deterministic:
        seed = 0
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, logger, writer)

def main_worker(gpu, ngpus_per_node, args, logger, writer):
    global best_acc1, its_ece
    args.gpu = gpu
    #     start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    model_teacher = models.resnet50(pretrained=False)
    model = reactnet()

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model_teacher.cuda(args.gpu)
            model.cuda(args.gpu)

            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_teacher = torch.nn.parallel.DistributedDataParallel(model_teacher, device_ids=[args.gpu])

        else:
            model_teacher.cuda(args.gpu)
            model.cuda(args.gpu)
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_teacher = torch.nn.parallel.DistributedDataParallel(model_teacher, device_ids=[args.gpu])
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model_teacher.cuda(args.gpu)
        model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model_teacher = torch.nn.DataParallel(model_teacher).cuda()
        model = torch.nn.DataParallel(model).cuda()

    # if os.path.exists(args.pretrained):
    #     # load model
    #     encoder_weight = torch.load(args.pretrained)['state_dict_model']
    #     classifier_weight = torch.load(args.pretrained)['state_dict_classifier']
    #     model_teacher.load_state_dict(encoder_weight, strict=False)
    #     model_teacher.load_state_dict(classifier_weight, strict=False)
    #     print(model_teacher.state_dict())
    # else:
    #     print('wrong pretrained path: ', args.pretrained)
    #     print('train from scratch')
    #     return

    # freeze teacher model
    # for p in model_teacher.parameters():
    #     p.requires_grad = False
    # model_teacher.eval()

    if args.dataset == 'imagenet_lt':
        dataset = ImageNet_LT(args.distributed, root=args.data_path,
                              batch_size=args.batch_size, num_works=args.workers)
    train_loader = dataset.train_instance
    val_loader = dataset.eval
    if args.distributed:
        train_sampler = dataset.dist_sampler

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)


if __name__ == '__main__':
    main()
