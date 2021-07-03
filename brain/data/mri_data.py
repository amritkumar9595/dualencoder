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
from pathlib import Path



class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root, acc_factor,dataset_type,mask_path): # acc_factor can be passed here and saved as self variable
    def __init__(self, root, acc_factor,dataset_type,sample_rate): # acc_factor can be passed here and saved as self variable
        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.acc_factor = 4 #int(acc_factor[-2]) 
        self.dataset_type = dataset_type
        
        self.sample_rate = sample_rate #   0.6
        
        random.shuffle(files)
        num_files = round(len(files) * self.sample_rate)
        files = files[:num_files]
        
        # self.key_img = 'img_volus_{}'.format(self.acc_factor)
        # self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor)
        # self.centre_fraction=[0.08]
        # self.accelaration = []
        
        #mask_path = os.path.join(mask_path,'mask_{}.npy'.format(acc_factor))
        #self.mask = np.load(mask_path)
        # self.accelaration.append(int(self.acc_factor[0]))
        # print("self_acc",self.accelaration)
        for fname in sorted(files):
            kspace = np.load(fname)#, allow_pickle=True)
            num_slices = kspace.shape[0]
            self.examples += [(fname, slice) for slice in range(20,num_slices-20)]   #20 20


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice = self.examples[i] 
        # Print statements 
        #print (fname,slice)

        data = np.load(fname)
        kspace = data[slice]
        kspace_cplx = kspace[:,:,0] + 1j*kspace[:,:,1]

        kspace = np.fft.fftshift(kspace_cplx)
        target = np.fft.ifft2(kspace_cplx)
        target_abs = np.abs(target)
        target_abs = target_abs/np.max(target_abs)
        
        kspace_cmplx = np.fft.fftshift(np.fft.fft2(target_abs,norm='ortho'))
        kspace = transforms.to_tensor(kspace_cmplx)
        
        # seed = None if not use_seed else tuple(map(ord, file))
        # print("acc=",self.acc_factor)
        mask_func = MaskFunc([0.08], [self.acc_factor])
       
        seed =  tuple(map(ord, str(fname)))
        
        # mask = MaskFunc([0.08], [4])
        masked_kspace, mask = transforms.apply_mask(kspace.float(),mask_func,seed)
        
        masked_kspace_np = masked_kspace[:,:,0].numpy() + 1j*masked_kspace[:,:,1].numpy()
        us_img = np.abs(np.fft.ifft2(masked_kspace_np))
        
        
        
        
        # target = data['volfs'][:,:,slice]

        # kspace_cmplx = np.fft.fftshift(np.fft.fft2(target,norm='ortho'))
        # kspace = transforms.to_tensor(kspace_cmplx)
        
        
        # mask_func = MaskFunc([0.08], self.accelaration)

        # seed =  tuple(map(ord, str(fname)))
        # masked_kspace_square, mask = transforms.apply_mask(kspace.float(), mask_func, seed)
        # masked_kspace_np = masked_kspace_square[:,:,0].numpy() + 1j*masked_kspace_square[:,:,1].numpy()
        # us_img = np.abs( np.fft.ifft2(masked_kspace_np))
            
            
            
            
            
        
            #uskspace_cmplx = kspace_cmplx * self.mask
            #zf_img = np.abs(np.fft.ifft2(uskspace_cmplx,norm='ortho'))
            
            # if self.dataset_type == 'cardiac':
                # Cardiac dataset should be padded,150 becomes 160. # this can be commented for kirby brain 
                # input_kspace is not padded, dont bother, as we are not using it "
        # print("masked_kspace",masked_kspace.shape)
        # masked_kspace = np.pad(masked_kspace,((2,2),(2,2),(0,0)),'constant',constant_values=(0,0))
        #     # print("masked_kspace2",masked_kspace_square.shape)
        # us_img  = np.pad(us_img,(2,2),'constant',constant_values=(0,0))
        # target_abs = np.pad(target_abs,(2,2),'constant',constant_values=(0,0))

            # Print statements
            #print (input.shape,target.shape)
            #return torch.from_numpy(zf_img), torch.from_numpy(target)
        
            
       
        fname = Path(fname)    
        return us_img, masked_kspace , target_abs , str(fname.name) , slice
            
