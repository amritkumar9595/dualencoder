import sys
import logging
import pathlib
import random
import shutil
import time
import functools
import numpy as np
import argparse

import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset import SliceData
from models import dAUTOMAP,UnetModelParallelEncoder, dAUTOMAPDualEncoderUnet
import torchvision
from torch import nn
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_datasets(args):
    
    train_mask = MaskFunc(args.center_fractions, args.accelerations)
    dev_mask = MaskFunc(args.center_fractions, args.accelerations)

    train_data = SliceData(
        root=args.train_path , 
        transform=DataTransform(train_mask, args.resolution),
        sample_rate=args.sample_rate
    )
    dev_data = SliceData(
        root=args.valid_path ,
        transform=DataTransform(dev_mask, args.resolution, use_seed=True),
        sample_rate=args.sample_rate,

    )
    return dev_data, train_data



def create_data_loaders(args):
    dev_data, train_data = create_datasets(args)

    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        #num_workers=64,
        #pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        #num_workers=64,
        #pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=16,
        #num_workers=64,
        #pin_memory=True,
    )
    return train_loader, dev_loader, display_loader


def train_epoch(args, epoch, model,data_loader, optimizer, writer):
    
    model.train()
    avg_loss = 0.0
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    #print ("Entering Train epoch")

    for iter, data in enumerate(tqdm(data_loader)):

        #print (data)

        #print ("Received data from loader")
        input,input_kspace,target = data # Return kspace also we can ignore that for train and test 
        # print("input",input.shape)
        # print("input_kspace",input_kspace.shape)
        # print("target",target.shape)

        input = input.float()
        input_kspace = input_kspace.float()
        target = target.float()

        input = input.unsqueeze(1).to(args.device)
        input_kspace = input_kspace.permute(0,3,1,2).to(args.device)
        target = target.unsqueeze(1).to(args.device)

        # dautomap_pred = dautomap_model(input_kspace)
        # dautomap_pred = F.pad(dautomap_pred,(5,5,5,5),"constant",0)

        output = model(input_kspace,input)

        loss = F.l1_loss(output,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('TrainLoss',loss.item(),global_step + iter )


        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
        #break

    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model,data_loader, writer):

    model.eval()
    losses = []
    start = time.perf_counter()
    
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
    
            input,input_kspace, target = data # Return kspace also we can ignore that for train and test

            input = input.float()
            input_kspace = input_kspace.float()
            target = target.float()

            input = input.unsqueeze(1).to(args.device)
            input_kspace = input_kspace.permute(0,3,1,2).to(args.device)
            target = target.unsqueeze(1).to(args.device)

            # dautomap_pred = dautomap_model(input_kspace)
            # dautomap_pred = F.pad(dautomap_pred,(5,5,5,5),"constant",0) # For cnn this might not be required. 

            output = model(input_kspace,input)
            #loss = F.mse_loss(output,target, size_average=False)
            loss = F.mse_loss(output,target)
            
            losses.append(loss.item())
            
        writer.add_scalar('Dev_Loss',np.mean(losses),epoch)
       
    return np.mean(losses), time.perf_counter() - start


def visualize(args, epoch, model, data_loader, writer):
    
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            input,input_kspace,target = data # Return kspace also we can ignore that for train and test

            input = input.float()
            input_kspace = input_kspace.float()
            target = target.float()

            input = input.unsqueeze(1).to(args.device)
            input_kspace = input_kspace.permute(0,3,1,2).to(args.device)
            target = target.unsqueeze(1).to(args.device)
            
            # dautomap_pred = dautomap_model(input_kspace)
            # dautomap_pred = F.pad(dautomap_pred,(5,5,5,5),"constant",0)

            output = model(input_kspace,input)

            #print("input: ", torch.min(input), torch.max(input))
            #print("target: ", torch.min(target), torch.max(target))
            #print("predicted: ", torch.min(output), torch.max(output))
            save_image(input, 'Input')
            # save_image(dautomap_pred,'dautomap output')
            save_image(output, 'Reconstruction')
            save_image(target, 'Target')
            save_image(torch.abs(target.float() - output.float()), 'Error')
            break

def save_model(args, exp_dir, epoch, model, optimizer,best_dev_loss,is_new_best):

    out = torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir':exp_dir
        },
        f=exp_dir / 'model.pt'
    )

    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')

def build_dualencoderunet(args):
    # print("device",args.device)
    model = UnetModelParallelEncoder(
        in_chans=1,
        out_chans=1,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob
    ).to(args.device)
    
    return model

def build_dautomap(args):
    # checkpoint = torch.load(checkpoint_file)
    # args = checkpoint['args']

    patch_size = 150
    model_params = {
      'input_shape': (2, patch_size, patch_size),
      'output_shape': (1, patch_size, patch_size),
      'tfx_params': {
        'nrow': patch_size,
        'ncol': patch_size,
        'nch_in': 2,
        'kernel_size': 1,
        'nl': 'relu',
        'init_fourier': False,
        'init': 'xavier_uniform_',
        'bias': True,
        'share_tfxs': False,
        'learnable': True,
      },
      'depth': 2,
      'nl':'relu'
    }

    model = dAUTOMAP(model_params['input_shape'],model_params['output_shape'],model_params['tfx_params']).to(args.device)

    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    # model.load_state_dict(checkpoint['model'])
    return model

# def load_unet(checkpoint_file):
#     checkpoint = torch.load(checkpoint_file)
#     args = checkpoint['args']
#     model = UnetModel(1, 1, args.num_chans, args.num_pools, args.drop_prob).to(args.device)
#     #print(model)
#     if args.data_parallel:
#         model = torch.nn.DataParallel(model)
#     model.load_state_dict(checkpoint['model'])
#     return model

def build_model(args):
    dautomap_model = build_dautomap(args)
    dualencoderunet_model = build_dualencoderunet(args)
    model = dAUTOMAPDualEncoderUnet(dautomap_model,dualencoderunet_model).to(args.device)

    return model


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    
    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args,model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint, model, optimizer



def build_optim(args, params):
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    return optimizer


def main(args):

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    #writer = SummaryWriter(logdir=str(args.exp_dir / 'summary'))
    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))
    
    if args.resume:
        print('resuming model , batch_size', args.batch_size)
        checkpoint,model,optimizer = load_model(args.checkpoint)
        args = checkpoint['args']
        args.batch_size =28
        best_dev_mse = checkpoint['best_dev_mse']
        best_dev_ssim = checkpoint['best_dev_mse'] ###why mse here ??
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model = build_model(args)
        print ("Model Built")
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args,model.parameters())
        print ("Optmizer initialized")
        best_dev_loss = 1e9
        start_epoch = 0
    
    logging.info(args)
    logging.info(model)


    train_loader, dev_loader, display_loader = create_data_loaders(args)
    print ("Dataloader initialized")


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    
    for epoch in range(start_epoch, args.num_epochs):

        scheduler.step(epoch)
        train_loss,train_time = train_epoch(args, epoch, model,train_loader,optimizer,writer)
        dev_loss,dev_time = evaluate(args, epoch, model,dev_loader, writer)
        visualize(args, epoch, model,display_loader, writer)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss,dev_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer,best_dev_loss,is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g}'
            f'DevLoss= {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    writer.close()


def create_arg_parser():

    parser = argparse.ArgumentParser(description='Train setup for MR recon U-Net')
    parser.add_argument('--seed',default=42,type=int,help='Seed for random number generators')
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')
    parser.add_argument('--batch-size', default=2, type=int,  help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true', 
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='storF.pad(e_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--train-path',type=str,help='Path to train h5 files')
    parser.add_argument('--validation-path',type=str,help='Path to test h5 files')

    parser.add_argument('--acceleration_factor',type=str,help='acceleration factors')
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby')
    # parser.add_argument('--dautomap_model_path',type=str,help='dautomap best model path')
    #parser.add_argument('--usmask_path',type=str,help='us mask path')
    
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print (args)
    main(args)
