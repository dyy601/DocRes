import os
import cv2
import time
import random
import datetime
import argparse
import numpy as np
from tqdm import tqdm

from itertools import cycle

import torch
import torch.nn as nn
from torch.utils import data
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from pathlib import Path

def mkdir(path): Path(path).mkdir(parents=True, exist_ok=True)
def get_lr(opt): return opt.param_groups[0]['lr']
def dict2string(d): return str(d)
def second2hours(t): return str(datetime.timedelta(seconds=int(t)))
from loaders import docres_loader
from models import restormer_arch


def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)
# seed_torch()

def getBasecoord(h,w):
    base_coord0 = np.tile(np.arange(h).reshape(h,1),(1,w)).astype(np.float32)
    base_coord1 = np.tile(np.arange(w).reshape(1,w),(h,1)).astype(np.float32)
    base_coord = np.concatenate((np.expand_dims(base_coord1,-1),np.expand_dims(base_coord0,-1)),-1)
    return base_coord

def train(args):
    if args.im_size < 8 or args.im_size % 8 or min(args.batch_size,args.total_iter,args.save_every) < 1 or args.workers < 0:
        raise ValueError('Invalid image size, batch size, iterations, save interval or workers')
    ## Single GPU init
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        raise RuntimeError('Use python train.py directly for single-GPU training.')
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda',args.local_rank)
    torch.cuda.manual_seed_all(42)
    ### Log file:
    mkdir(args.logdir)
    mkdir(os.path.join(args.logdir,args.experiment_name))
    log_file_path=os.path.join(args.logdir,args.experiment_name,'log.txt')
    log_file=open(log_file_path,'a')
    log_file.write('\n---------------  '+args.experiment_name+'  ---------------\n')
    log_file.close()

    ### Setup tensorboard for visualization
    if args.tboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.logdir,args.experiment_name,'runs'),args.experiment_name)
    ### Setup Dataloader
    # 去模糊、去阴影、几何形变、二值化和外观增强五个任务的数据集配置；
    # 每项包含任务名、采样比率、图像路径和 JSON 标签路径。
    data_root = Path(args.data_root)
    datasets_setting = [dict(task=t, ratio=1, im_path=str(data_root/t), json_paths=[str(data_root/t/'train.json')]) for t in ['deblurring','dewarping','binarization','deshadowing','appearance']]

    ratios = [dataset_setting['ratio'] for dataset_setting in datasets_setting]
    datasets = [docres_loader.DocResTrainDataset(dataset=dataset_setting,img_size=args.im_size) for dataset_setting in datasets_setting]
    trainloaders = []
    for setting, dataset in zip(datasets_setting, datasets):
        if len(dataset) < args.batch_size:
            raise ValueError(f"Dataset {setting['task']} is smaller than batch_size")
        loader = data.DataLoader(dataset, shuffle=True, batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=True, drop_last=True)
        trainloaders.append({'task':setting, 'loader':loader, 'iter_loader':iter(loader)})

    ### test loader
    # for i in tqdm(range(args.total_iter)):
    #     loader_index = iters % len(trainloaders) if args.task_cycle else random.choices(list(range(len(trainloaders))),ratios)[0]
    #     in_im,gt_im = next(trainloaders[loader_index]['iter_loader'])

    ### Setup Model
    model = restormer_arch.Restormer(
        inp_channels=6,
        out_channels=3,
        dim = 48,
        num_blocks = [2,3,3,4],
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',
        dual_pixel_task = True
    )
    model=model.to(device)
    ### Optimizer
    optimizer= torch.optim.AdamW(model.parameters(),lr=args.l_rate,weight_decay=5e-4)

    ### LR Scheduler
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.total_iter, eta_min=1e-6, last_epoch=-1)
    ### load checkpoint
    iter_start=0
    if args.resume is not None:
        print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        x = {k.removeprefix('module.'):v for k,v in checkpoint['model_state'].items()}
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'scheduler_state' in checkpoint: sched.load_state_dict(checkpoint['scheduler_state'])
        else: sched.last_epoch = checkpoint.get('iters', checkpoint.get('iter', 0))
        model.load_state_dict(x,strict=True)
        iter_start=checkpoint.get('iters', checkpoint.get('iter', 0))
        print("Loaded checkpoint '{}' (iter {})".format(args.resume, iter_start))
    ###-----------------------------------------Training-----------------------------------------
    ##initialize
    scaler = torch.amp.GradScaler('cuda')
    if args.resume and 'scaler_state' in checkpoint: scaler.load_state_dict(checkpoint['scaler_state'])
    loss_dict = {}
    total_step = 0
    l2 = nn.MSELoss()
    l1 = nn.L1Loss()
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    m = nn.Sigmoid()
    best = 0
    best_ce = 999
    ## total_steps
    for iters in range(iter_start,args.total_iter):
        start_time = time.time()
        loader_index = iters % len(trainloaders) if args.task_cycle else random.choices(list(range(len(trainloaders))),ratios)[0]
        try:
            in_im,gt_im = next(trainloaders[loader_index]['iter_loader'])
        except StopIteration:
            trainloaders[loader_index]['iter_loader']=iter(trainloaders[loader_index]['loader'])
            in_im,gt_im = next(trainloaders[loader_index]['iter_loader'])
        in_im = in_im.float().cuda()
        gt_im = gt_im.float().cuda()
        binarization_loss,appearance_loss,dewarping_loss,deblurring_loss,deshadowing_loss = 0,0,0,0,0
        with torch.autocast('cuda'):
            pred_im = model(in_im,trainloaders[loader_index]['task']['task'])
            if trainloaders[loader_index]['task']['task'] == 'binarization':
                gt_im = gt_im.long()
                binarization_loss = ce(pred_im[:,:2,:,:], gt_im[:,0,:,:])
                loss = binarization_loss
            elif trainloaders[loader_index]['task']['task'] == 'dewarping':
                dewarping_loss = l1(pred_im[:,:2,:,:], gt_im[:,:2,:,:])
                loss = dewarping_loss
            elif trainloaders[loader_index]['task']['task'] == 'appearance':
                appearance_loss = l1(pred_im, gt_im)
                loss = appearance_loss
            elif trainloaders[loader_index]['task']['task'] == 'deblurring':
                deblurring_loss = l1(pred_im, gt_im)
                loss = deblurring_loss
            elif trainloaders[loader_index]['task']['task'] == 'deshadowing':
                deshadowing_loss = l1(pred_im, gt_im)
                loss = deshadowing_loss
        if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
        optimizer.zero_grad(set_to_none=True)
        previous_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= previous_scale: sched.step()
        total_step = iters+1

        loss_dict['dew_loss']=dewarping_loss.item() if isinstance(dewarping_loss,torch.Tensor) else 0
        loss_dict['app_loss']=appearance_loss.item() if isinstance(appearance_loss,torch.Tensor) else 0
        loss_dict['des_loss']=deshadowing_loss.item() if isinstance(deshadowing_loss,torch.Tensor) else 0
        loss_dict['deb_loss']=deblurring_loss.item() if isinstance(deblurring_loss,torch.Tensor) else 0
        loss_dict['bin_loss']=binarization_loss.item() if isinstance(binarization_loss,torch.Tensor) else 0
        end_time = time.time()
        duration = end_time-start_time
        ## log
        if (iters+1) % 10 == 0 or args.task_cycle:
            ## print
            print('iters [{}/{}] -- '.format(iters+1,args.total_iter)+dict2string(loss_dict)+' --lr {:6f}'.format(get_lr(optimizer))+' -- time {}'.format(second2hours(duration*(args.total_iter-iters))))
            ## tbord
            if args.tboard:
                for key,value in loss_dict.items():
                    writer.add_scalar('Train '+key+'/Iterations', value, total_step)
            ## logfile
            with open(log_file_path,'a') as f:
                f.write('iters [{}/{}] -- '.format(iters+1,args.total_iter)+dict2string(loss_dict)+' --lr {:6f}'.format(get_lr(optimizer))+' -- time {}'.format(second2hours(duration*(args.total_iter-iters)))+'\n')

        if (iters+1) % args.save_every == 0 or iters+1 == args.total_iter:
            state = {'iters': iters+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(), 'scheduler_state':sched.state_dict(), 'scaler_state':scaler.state_dict()}
            if not os.path.exists(os.path.join(args.logdir,args.experiment_name)):
                 os.system('mkdir ' + os.path.join(args.logdir,args.experiment_name))
            if True:
                torch.save(state, os.path.join(args.logdir,args.experiment_name,"{}.pkl".format(iters+1)))
    if args.tboard: writer.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--im_size', nargs='?', type=int, default=256,
                        help='Height of the input image')
    parser.add_argument('--total_iter', nargs='?', type=int, default=100000,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=2e-4,
                        help='Learning Rate')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--logdir', nargs='?', type=str, default='./checkpoints/',
                        help='Path to store the loss logs')
    parser.add_argument('--tboard', dest='tboard', action='store_true',
                        help='Enable visualization(s) on tensorboard | False by default')
    parser.add_argument('--local_rank',type=int,default=0,metavar='N')
    parser.add_argument('--experiment_name', nargs='?', type=str,default='experiment_name',
                        help='the name of this experiment')
    parser.add_argument('--data_root', default=str(Path(__file__).resolve().parent/'data/train'))
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--task_cycle', action='store_true')
    parser.set_defaults(tboard=False)
    args = parser.parse_args()
    train(args)
