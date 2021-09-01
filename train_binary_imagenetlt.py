import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import pprint
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from datasets.cifar10 import CIFAR10_LT
from datasets.cifar100 import CIFAR100_LT
from datasets.places import Places_LT
from datasets.imagenet import ImageNet_LT
from datasets.ina2018 import iNa2018

from models import resnet
from models import resnet_places
from models import resnet_cifar

from utils import config, update_config, create_logger
from utils import AverageMeter, ProgressMeter
from utils import accuracy, calibration

from methods import mixup_data, mixup_criterion

from binary.reactnet_optimized import Reactnet


def parse_args():
    parser = argparse.ArgumentParser(description='MiSLAS training (Stage-1)')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)

    return args


best_acc1 = 0
its_ece = 100


def main():
    args = parse_args()
    logger, model_dir, writer = create_logger(config, args.cfg)
    logger.info('\n' + pprint.pformat(args))
    logger.info('\n' + str(config))

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

    if config.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    # Simply call main_worker function

    global best_acc1, its_ece

#     start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    if config.gpu is not None:
        logger.info("Use GPU: {} for training".format(config.gpu))

    model = Reactnet(num_classes=1000)

    if not torch.cuda.is_available():
        logger.info('using CPU, this will be slow')
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    elif torch.cuda.device_count() > 1:
        print('use %d gpus' %(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model).cuda()

    torch.backends.cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # optionally resume from a checkpoint
    cur_epoch = 0
    if config.resume:
        if os.path.isfile(config.resume):
            logger.info("=> loading checkpoint '{}'".format(config.resume))
            if config.gpu is None:
                checkpoint = torch.load(config.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(config.gpu)
                checkpoint = torch.load(config.resume, map_location=loc)
            config.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if config.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(config.gpu)
            model.load_state_dict(checkpoint['state_dict_model'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(config.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.resume))

    dataset = ImageNet_LT(config.distributed, root=config.data_path,
                          batch_size=config.batch_size, num_works=config.workers)

    train_loader = dataset.train_instance
    val_loader = dataset.eval

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

    for epoch in range(config.num_epochs):
        # adjust_learning_rate(optimizer, scheduler, epoch, config)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, scheduler, epoch, config, logger, writer)

        # evaluate on validation set
        acc1, loss, ece = validate(val_loader, model, criterion, config, logger)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            its_ece = ece
        logger.info('Best Prec@1: %.3f%% ECE: %.3f%%\n' % (best_acc1, its_ece))
        writer.add_scalar('val loss', loss, epoch)
        writer.add_scalar('val ece', ece, epoch)
        writer.add_scalar('val acc', acc1, epoch)

        if epoch % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict_model': model.state_dict(),
                'best_acc1': best_acc1,
                'its_ece': its_ece,
            }, is_best, model_dir)
    writer.close()


def train(train_loader, model, criterion, optimizer, scheduler, epoch, config, logger, writer=None):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    training_data_num = len(train_loader.dataset)
    end_steps = int(training_data_num / train_loader.batch_size)

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        if i > end_steps:
            break

        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda()
            target = target.cuda()

        if config.mixup is True:
            images, targets_a, targets_b, lam = mixup_data(images, target, alpha=config.alpha)
            output = model(images)
            loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
        else:

            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            progress.display(i, logger)
        if writer:
            writer.add_scalar('train loss', losses.avg, epoch)
            writer.add_scalar('train acc', top1.avg, epoch)


def validate(val_loader, model, criterion, config, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Eval: ')

    # switch to evaluate mode
    model.eval()

    class_num = torch.zeros(config.num_classes).cuda()
    correct = torch.zeros(config.num_classes).cuda()

    confidence = np.array([])
    pred_class = np.array([])
    true_class = np.array([])

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda()
                target = target.cuda()
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            _, predicted = output.max(1)
            target_one_hot = F.one_hot(target, config.num_classes)
            predict_one_hot = F.one_hot(predicted, config.num_classes)
            class_num = class_num + target_one_hot.sum(dim=0).to(torch.float)
            correct = correct + (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)

            prob = torch.softmax(output, dim=1)
            confidence_part, pred_class_part = torch.max(prob, dim=1)
            confidence = np.append(confidence, confidence_part.cpu().numpy())
            pred_class = np.append(pred_class, pred_class_part.cpu().numpy())
            true_class = np.append(true_class, target.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                progress.display(i, logger)

        acc_classes = correct / class_num
        head_acc = acc_classes[config.head_class_idx[0]:config.head_class_idx[1]].mean() * 100

        med_acc = acc_classes[config.med_class_idx[0]:config.med_class_idx[1]].mean() * 100
        tail_acc = acc_classes[config.tail_class_idx[0]:config.tail_class_idx[1]].mean() * 100
        logger.info(
            '* Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}% HAcc {head_acc:.3f}% MAcc {med_acc:.3f}% TAcc {tail_acc:.3f}%.'.format(
                top1=top1, top5=top5, head_acc=head_acc, med_acc=med_acc, tail_acc=tail_acc))

        cal = calibration(true_class, pred_class, confidence, num_bins=15)
        logger.info('* ECE   {ece:.3f}%.'.format(ece=cal['expected_calibration_error'] * 100))

    return top1.avg, losses.avg, cal['expected_calibration_error'] * 100

def save_checkpoint(state, is_best, model_dir):
    filename = model_dir + '/current.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, model_dir + '/model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, config):
    """Sets the learning rate"""
    if config.cos:
        lr_min = 0
        lr_max = config.lr
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(epoch / config.num_epochs * 3.1415926535))
    else:
        epoch = epoch + 1
        if epoch <= 5:
            lr = config.lr * epoch / 5
        elif epoch > 180:
            lr = config.lr * 0.01
        elif epoch > 160:
            lr = config.lr * 0.1
        else:
            lr = config.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_writer(cfg, model_dir):
    # time_str = time.strftime('%Y%m%d%H%M')
    log_dir = model_dir.replace('ckps', 'logs')
    print('=> creating {}'.format(log_dir))
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)
    return writer


if __name__ == '__main__':
    main()