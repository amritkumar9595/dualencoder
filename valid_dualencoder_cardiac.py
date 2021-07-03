import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.mri_data import SliceData
from models.models import dAUTOMAP , UnetModelParallelEncoder, dAUTOMAPDualEncoderUnet
import h5py
from tqdm import tqdm

def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.
    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir.mkdir(exist_ok=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)


def create_data_loaders(args):

    #data = SliceDataDev(args.data_path,args.acceleration_factor,args.dataset_type,args.usmask_path)
    # data = SliceDataDev(args.data_path,args.acceleration_factor,args.dataset_type)
    data = SliceData(args.data_path,args.acceleration_factor,args.dataset_type)

    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )

    return data_loader



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

    patch_size =  160 #args.resolution
    model_params = {
      'input_shape': (2, patch_size, patch_size),
      'output_shape': (1, patch_size, patch_size),
      'tfx_params': {
        'nrow': patch_size,
        'ncol': patch_size,
        'nch_in': 2,
        'kernel_size': 1,
        'nl': None,
        'init_fourier': True,
        'init': 'xavier_uniform_',
        'bias': False, #True,
        'share_tfxs': False,
        'learnable': True,
        'shift': False
      },
      'tfx_params2': {
        'nrow': patch_size,
        'ncol': patch_size,
        'nch_in': 2,
        'kernel_size': 1,
        'nl': 'relu',
        'init_fourier': False,
        'init': 'xavier_uniform_',
        'bias':True,
        'share_tfxs': False,
        'learnable': True,
        'shift': False
      },
      'depth': 2,
      'nl':'relu'
    }

    model = dAUTOMAP(model_params['input_shape'],model_params['output_shape'],model_params['tfx_params'],model_params['tfx_params2']).to(args.device)
    
    return model


def build_model(args):
    dautomap_model = build_dautomap(args)
    dualencoderunet_model = build_dualencoderunet(args)
    model = dAUTOMAPDualEncoderUnet(dautomap_model,dualencoderunet_model).to(args.device)
    
    # model = dautomap_model
    return model


def load_model(checkpoint_file):
    
    print("model loaded from :",checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']

    dautomap_model = build_dautomap(args)
    dualencoderunet_model = build_dualencoderunet(args)
    model = dAUTOMAPDualEncoderUnet(dautomap_model,dualencoderunet_model).to(args.device)
    
    model.load_state_dict(checkpoint['model'])
    return model


def run_model(args, model, data_loader):

    model.eval()
    reconstructions = defaultdict(list)

    with torch.no_grad():

        for (iter,data) in enumerate(tqdm(data_loader)):

            us_img,inp_kspace, target,fnames,slices = data
            #input = input.unsqueeze(1).to(args.device)
            inp_kspace = inp_kspace.permute(0,3,1,2)
            inp_kspace = inp_kspace.float().to(args.device)
            
            us_img = us_img.float().unsqueeze(1).to(args.device)
            
            recons,_ = model(inp_kspace,us_img)
            recons = recons.to('cpu').squeeze(1)
            # cardiac mri crop to 150,150
            if args.dataset_type == 'cardiac':
                recons = recons[:,5:155,5:155]
            
            for i in range(recons.shape[0]):
                recons[i] = recons[i] 
                reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }
    return reconstructions


def main(args):
    
    data_loader = create_data_loaders(args)
    model = load_model(args.checkpoint)
    reconstructions = run_model(args, model, data_loader)
    save_reconstructions(reconstructions, args.out_dir)
    print("reconstructions @:",args.out_dir)


def create_arg_parser():

    parser = argparse.ArgumentParser(description="Valid setup for MR recon U-Net")
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    parser.add_argument('--data-path',type=str,help='path to validation dataset')

    parser.add_argument('--acceleration_factor',type=str,help='acceleration factors')
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby')
    #parser.add_argument('--usmask_path',type=str,help='undersampling mask path')

    
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
