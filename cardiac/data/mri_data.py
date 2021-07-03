"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import pathlib
import random
from data import transforms
import h5py
from torch.utils.data import Dataset
import numpy as np

from common.subsample import MaskFunc



class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root, acc_factor,dataset_type,mask_path): # acc_factor can be passed here and saved as self variable
    def __init__(self, root, acc_factor,dataset_type): # acc_factor can be passed here and saved as self variable
        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.acc_factor = acc_factor 
        self.dataset_type = dataset_type
        # self.key_img = 'img_volus_{}'.format(self.acc_factor)
        # self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor)
        # self.centre_fraction=[0.08]
        self.accelaration = []
        
        #mask_path = os.path.join(mask_path,'mask_{}.npy'.format(acc_factor))
        #self.mask = np.load(mask_path)
        self.accelaration.append(int(self.acc_factor[0]))
        # print("self_acc",self.accelaration)
        for fname in sorted(files):
            # print("fname",fname)
            with h5py.File(fname,'r') as hf:
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice = self.examples[i] 
        # Print statements 
        #print (fname,slice)
    
        with h5py.File(fname, 'r') as data:

            # input_img  = data[self.key_img][:,:,slice]
            # input_kspace  = data[self.key_kspace][:,:,slice]
            # input_kspace = npComplexToTorch(input_kspace)
    
            target = data['volfs'][:,:,slice]

            kspace_cmplx = np.fft.fftshift(np.fft.fft2(target,norm='ortho'))
            kspace = transforms.to_tensor(kspace_cmplx)
            
            
            mask_func = MaskFunc([0.08], self.accelaration)

            seed =  tuple(map(ord, str(fname)))
            masked_kspace_square, mask = transforms.apply_mask(kspace.float(), mask_func, seed)
            masked_kspace_np = masked_kspace_square[:,:,0].numpy() + 1j*masked_kspace_square[:,:,1].numpy()
            us_img = np.abs( np.fft.ifft2(masked_kspace_np))
            
            
            
            
            
            
            #uskspace_cmplx = kspace_cmplx * self.mask
            #zf_img = np.abs(np.fft.ifft2(uskspace_cmplx,norm='ortho'))
            
            # if self.dataset_type == 'cardiac':
                # Cardiac dataset should be padded,150 becomes 160. # this can be commented for kirby brain 
                # input_kspace is not padded, dont bother, as we are not using it "
            # print("masked_kspace",masked_kspace_square.shape)
            masked_kspace_square = np.pad(masked_kspace_square,((5,5),(5,5),(0,0)),'constant',constant_values=(0,0))
            # print("masked_kspace2",masked_kspace_square.shape)
            us_img  = np.pad(us_img,(5,5),'constant',constant_values=(0,0))
            target = np.pad(target,(5,5),'constant',constant_values=(0,0))

            # Print statements
            #print (input.shape,target.shape)
            #return torch.from_numpy(zf_img), torch.from_numpy(target)
            # print("fname",type(fname))
            return us_img, masked_kspace_square , target , str(fname.name) , slice
            
