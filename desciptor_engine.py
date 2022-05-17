import argparse
import builtins
import os
import random
import shutil
import warnings
from pathlib import Path
import json
import numpy as np
import pandas as pd
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from pytorch_metric_learning import losses
from pytorch_metric_learning.utils import distributed as pml_dist
from tqdm import tqdm
from descriptor_dataset import ISCTrainDataset, InferenceDataset
from descriptor_net import ISCNet

warnings.simplefilter('ignore', UserWarning)
ver = __file__.replace('.py', '')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
parser.add_argument('-j', '--workers', default=os.cpu_count(), type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--gem-p', default=3.0, type=float)
parser.add_argument('--gem-eval-p', default=4.0, type=float)

parser.add_argument('--mode', default='train', type=str, help='train or extract')
parser.add_argument('--target-set', default='qr', type=str, help='q: query, r: reference, t: training')
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--pos-margin', default=0.0, type=float)
parser.add_argument('--neg-margin', default=0.7, type=float)
parser.add_argument('--ncrops', default=2, type=int)
parser.add_argument('--input-size', default=224, type=int)
parser.add_argument('--sample-size', default=100000, type=int)
parser.add_argument('--weight', type=str)
parser.add_argument('--eval-subset', action='store_true')
parser.add_argument('--memory-size', default=1024, type=int)
parser.add_argument('--tta', action='store_true')

def train(args):

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

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
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier(device_ids=[args.gpu])

    backbone = timm.create_model(args.arch, features_only=True, pretrained=True)
    model = ISCNet(backbone, p=args.gem_p, eval_p=args.gem_eval_p)

    if args.weight is not None:
        state_dict = torch.load(args.weight, map_location='cpu')['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('module.'):
                state_dict[k[len('module.'):]] = state_dict[k]
                del state_dict[k]
        model.load_state_dict(state_dict, strict=False)

    # infer learning rate before changing batch size
    init_lr = args.lr  # * args.batch_size / 256

    if args.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    loss_fn = losses.ContrastiveLoss(pos_margin=args.pos_margin, neg_margin=args.neg_margin)
    loss_fn = losses.CrossBatchMemory(loss_fn, embedding_size=256, memory_size=args.memory_size)
    loss_fn = pml_dist.DistributedLossWrapper(loss=loss_fn, device_ids=[args.rank])

    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or "gain" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    optim_params = [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': args.weight_decay}
    ]

    optimizer = torch.optim.SGD(optim_params, init_lr, momentum=args.momentum)
    scaler = torch.cuda.amp.GradScaler()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    gt = pd.read_csv('../input/public_ground_truth.csv')
    gt_ = gt[gt['reference_id'].notna()]
    query_paths = [f'/DATA/input/query_images/{qid}.jpg' for qid in gt_['query_id']]
    reference_paths = [f'/DATA/input/reference_images/{rid}.jpg' for rid in gt_['reference_id']]
    all_ref_ids = set([f'R{i:06d}' for i in range(1000000)])
    diff_ref_ids = all_ref_ids - set(gt_['reference_id'])
    diff_reference_paths = [f'/DATA/input/reference_images/{rid}.jpg' for rid in diff_ref_ids]

    preprocess = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=backbone.default_cfg['mean'], std=backbone.default_cfg['std'])
    ])

    train_dataset = ISCTrainDataset(
        query_paths,
        reference_paths,
        diff_reference_paths,
        preprocess,
        num_negatives=torch.cuda.device_count() * 2 - 2,
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        train_one_epoch(train_loader, model, loss_fn, optimizer, scaler, epoch, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
            }, is_best=False, filename=f'{ver}/train/checkpoint_{epoch:04d}.pth.tar')


def train_one_epoch(train_loader, model, loss_fn, optimizer, scaler, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    progress = tqdm(train_loader, desc=f'epoch {epoch}', leave=False, total=len(train_loader))

    model.train()

    for i, images, j in progress:
        optimizer.zero_grad()

        labels = torch.cat([torch.tile(i, dims=(2,)), torch.tensor(j)])
        labels = labels.cuda(args.gpu, non_blocking=True)
        images = torch.cat([
            image for image in images
        ], dim=0).cuda(args.gpu, non_blocking=True)

        with torch.cuda.amp.autocast():
            embeddings = model(images)
            loss = loss_fn(embeddings, labels)

        losses.update(loss.item(), images.size(0))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        progress.set_postfix(loss=losses.avg)

    print(f'epoch={epoch}, loss={losses.avg}')


# def extract(args):
def extract(json_path, data_dir, 
            weight="/home/kim/juche/projects/ISC21-Descriptor-Track-1st/exp/v107/train/checkpoint_0009.pth.tar"):
    
#     paths = sorted(Path("./").glob('../../DeFRCN/datasets/main/JPEGImages/*.jpg'))

#     ids = np.array([p.stem for p in paths], dtype='S6')

    backbone = timm.create_model('tf_efficientnetv2_m_in21ft1k', features_only=True, pretrained=True)
    model = ISCNet(backbone, p=3.0, eval_p=1.0)
    model = nn.DataParallel(model)

    state_dict = torch.load(weight, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict, strict=False)

    model.eval().cuda()

    cudnn.benchmark = True

    preprocesses = [
        transforms.Resize((int(512 * 1.4142135623730951), int(512 * 1.4142135623730951))),
        transforms.ToTensor(),
        transforms.Normalize(mean=backbone.default_cfg['mean'], std=backbone.default_cfg['std'])
    ]
    # print(backbone.default_cfg['mean'], backbone.default_cfg['std'])

    # specify path to json file
    dataset = InferenceDataset(json_path, data_dir, transforms=transforms.Compose(preprocesses))
    loader_kwargs = dict(batch_size=1, shuffle=False, num_workers=32, drop_last=False)
    data_loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)

    def calc_feats(loader):
        feats = []
        for _, image in tqdm(loader, total=len(loader)):
            image = image.cuda()
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_big = image
                image = F.interpolate(image_big, size=512, mode='bilinear', align_corners=False)
                image_small = F.interpolate(image, scale_factor=0.7071067811865475, mode='bilinear', align_corners=False)
                f = (
                    model(image) + model(image_small) + model(image_big)
                    + model(transforms.functional.hflip(image))
                    + model(transforms.functional.hflip(image_small))
                    + model(transforms.functional.hflip(image_big))
                )
                f /= torch.linalg.norm(f, dim=1, keepdim=True)
                
            feats.append(f.cpu().numpy())
        feats = np.concatenate(feats, axis=0)
        return feats.astype(np.float32)

    feats = calc_feats(data_loader)
    
    for i, f in enumerate(feats):
#         print("Feature:", f)
        dataset.data[i]["embedding"] = f.tolist()
#     with h5py.File(out, 'w') as f:
#         f.create_dataset('feats', data=feats)
#         f.create_dataset('ids', data=ids)
    out = os.path.splitext(dataset.json_path)[0] + ".embeddings" + ".json"
    with open(out, mode="w", encoding="utf-8") as f:
        json.dump(dataset.data, f)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * (1 - (epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    if not Path(f'{ver}/train').exists():
        Path(f'{ver}/train').mkdir(parents=True)
    if not Path(f'{ver}/extract').exists():
        Path(f'{ver}/extract').mkdir(parents=True)

    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'extract':
        extract(args)