#  Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# Python imports
import tqdm
from tqdm import tqdm
import os
from os.path import join as ospj
import csv

#Local imports
from data import dataset as dset
from models.common import Evaluator
from utils.utils import save_args, load_args
from utils.config_model import configure_model
from flags import parser, DATA_FOLDER
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from utils.utils import init_distributed_mode, is_main_process

import wandb

best_auc = 0
best_hm = 0

def main():
    # Get arguments and start logging
    args = parser.parse_args()
    init_distributed_mode(args)    
    device = torch.device(args.device)

    load_args(args.config, args)
    logpath = os.path.join(args.cv_dir, args.name)
    if is_main_process():
        os.makedirs(logpath, exist_ok=True)
        save_args(args, logpath, args.config)
        writer = SummaryWriter(log_dir = logpath, flush_secs = 30)
    else:
        writer = None

   
    # Get dataset
    trainset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='train',
        split=args.splitname,
        pair_dropout=args.pair_dropout,
        train_only= args.train_only,
        open_world=args.open_world,
        norm_family=args.norm_family,
        dataset=args.dataset
    )
    if args.distributed:
        sampler = DistributedSampler(trainset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True, drop_last=False)
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            sampler=sampler)
    else:
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers) 
    testset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase=args.test_set,
        split=args.splitname,
        subset=args.subset,
        open_world=args.open_world,
        norm_family=args.norm_family,
        dataset=args.dataset
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers)


    # Get model and optimizer
    model, optimizer = configure_model(args, trainset)

    train = train_normal

    evaluator_val =  Evaluator(testset, model)

    start_epoch = 0
    # Load checkpoint
    if args.load is not None:
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        print('Loaded model from ', args.load)
    
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
    if is_main_process():
        run = wandb.init(
            project="CAILA",
            config={
            })

    scaler = torch.cuda.amp.GradScaler()

    # create model and move it to GPU with id rank
    model = model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    
    for epoch in tqdm(range(start_epoch, args.max_epochs + 1), desc = 'Current epoch'):
        p_mixup = args.mixup_ratio
        trainloader.dataset.set_p(p_mixup, args.concept_shift_prob, args.obj_shift_ratio)
        if args.distributed:
            trainloader.sampler.set_epoch(epoch)
        train(epoch, args, model, trainloader, optimizer, writer, device, scaler)
        if is_main_process():
            if (epoch + 1) % args.eval_val_every == 0:
                with torch.no_grad(): # todo: might not be needed
                    test(epoch, model.module, testloader, evaluator_val, writer, args, logpath, device)
        dist.barrier()  
    
    print('Best AUC achieved is ', best_auc)
    print('Best HM achieved is ', best_hm)


def train_normal(epoch, args, model, trainloader, optimizer, writer, device, scaler):
    '''
    Runs training for an epoch
    '''

    model.train() # Let's switch to training

    train_loss = 0.0 
    for idx, data in tqdm(enumerate(trainloader), total=len(trainloader), desc = 'Training'):
        data  = [d.to(device) for d in data]

        with torch.cuda.amp.autocast(enabled=args.fp16):
            loss, _ = model(data)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
        train_loss += loss.item()

    train_loss = train_loss/len(trainloader)
    if is_main_process():
        writer.add_scalar('Loss/train_total', train_loss, epoch)
        wandb.log({'Train/loss_total': train_loss}, step=epoch)
    print('Epoch: {}| Loss: {}'.format(epoch, round(train_loss, 2)))


def test(epoch, model, testloader, evaluator, writer, args, logpath, device):
    '''
    Runs testing for an epoch
    '''
    global best_auc, best_hm

    def save_checkpoint(filename):
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
            'AUC': stats['AUC']
        }
        torch.save(state, os.path.join(logpath, 'ckpt_{}.t7'.format(filename)))

    model.eval()
    model.reset_saved_pair_embeds()

    accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []

    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
        data = [d.to(device) for d in data]

        _, predictions = model(data)
        
        predictions = predictions.cpu()

        attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]

        all_pred.append(predictions)
        all_attr_gt.append(attr_truth.cpu())
        all_obj_gt.append(obj_truth.cpu())
        all_pair_gt.append(pair_truth.cpu())

    if args.cpu_eval:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt), torch.cat(all_obj_gt), torch.cat(all_pair_gt)
    else:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
            'cpu'), torch.cat(all_pair_gt).to('cpu')
    
    print("Converted to CPU")

    # Calculate best unseen accuracy
    all_pred_dict = torch.cat(all_pred, dim=0)
    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=args.bias, topk=args.topk)
    print("Done Running Results")
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=args.topk)

    stats['a_epoch'] = epoch

    result = ''
    # write to Tensorboard
    for key in stats:
        writer.add_scalar(key, stats[key], epoch)
        wandb.log({'Val/{}'.format(key): stats[key]}, step=epoch)
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '

    result = result + args.name
    print(f'Test Epoch: {epoch}')
    print(result)
    if epoch > 0 and epoch % args.save_every == 0:
        save_checkpoint(epoch)
    if stats['AUC'] > best_auc:
        best_auc = stats['AUC']
        print('New best AUC ', best_auc)
        save_checkpoint('best_auc')

    if stats['best_hm'] > best_hm:
        best_hm = stats['best_hm']
        print('New best HM ', best_hm)
        save_checkpoint('best_hm')

    # Logs
    with open(ospj(logpath, 'logs.csv'), 'a') as f:
        w = csv.DictWriter(f, stats.keys())
        if epoch == 0:
            w.writeheader()
        w.writerow(stats)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Best AUC achieved is ', best_auc)
        print('Best HM achieved is ', best_hm)
