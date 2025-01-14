from pathlib import Path
from yacs.config import CfgNode as CN
import os
import time
import logging
from torch.utils.tensorboard import SummaryWriter

_C = CN()
_C.name = ''
_C.print_freq = 40
_C.workers = 2
_C.log_dir = 'logs'
_C.model_dir = 'ckps'
_C.simsiam = None
_C.binary = False

_C.temperature = 3.0
_C.alpha = 0.1
_C.beta = 1e-6

_C.dataset = 'cifar10'
_C.data_path = './data/cifar10'
_C.num_classes = 100
_C.imb_factor = 0.01
_C.backbone = 'resnet32_fe'
_C.resume = ''
_C.head_class_idx = [0, 1]
_C.med_class_idx = [0, 1]
_C.tail_class_idx = [0, 1]

_C.deterministic = True
_C.gpu = 0
_C.world_size = -1
_C.rank = -1
_C.dist_url = 'tcp://224.66.41.62:23456'
_C.dist_backend = 'nccl'
_C.multiprocessing_distributed = False
_C.distributed = False

_C.mode = None
_C.smooth_tail = None
_C.smooth_head = None
_C.shift_bn = False
_C.lr_factor = None
_C.lr = 0.1
_C.optimizer = 'adam'
_C.lr_scheduler = 'cosine'
_C.batch_size = 128
_C.weight_decay = 0.0
_C.num_epochs = 200
_C.momentum = 0.9
_C.cos = False
_C.mixup = False
_C.alpha = 0.0
_C.mixup = True
_C.alpha = 1.0



def update_config(cfg, args):
    cfg.defrost()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    # cfg.freeze()


def create_logger(cfg, cfg_name, add_date=False):
    # time_str = time.strftime('%Y%m%d%H%M')

    cfg_name = os.path.basename(cfg_name).split('.')[0]

    path = os.path.join('saved', 
                        cfg_name, 
                        (cfg.dataset + '_'+ (str)(cfg.imb_factor)),                        
                    )
    # if add_date:
    #     log_dir = Path("saved") / (cfg_name + '_' + time_str) / Path(cfg.log_dir)
    # else:
    log_dir = os.path.join(path, 'logs')
    print('=> creating {}'.format(log_dir))
    os.makedirs(log_dir, exist_ok=True)

    log_file = '{}.txt'.format(cfg_name)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    writer = SummaryWriter(log_dir)

    model_dir = os.path.join(path, 'ckps')
    print('=> creating {}'.format(model_dir))
    os.makedirs(model_dir, exist_ok=True)

    return logger, str(model_dir), writer

