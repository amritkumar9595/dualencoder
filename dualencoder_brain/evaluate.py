import argparse
import pathlib
from argparse import ArgumentParser

import h5py
import numpy as np
from runstats import Statistics
from skimage.measure import compare_psnr, compare_ssim
from skimage.filters import laplace
from tqdm import tqdm

# adding hfn metric 
def hfn(gt,pred):

    hfn_total = []

    for ii in range(gt.shape[-1]):
        gt_slice = gt[:,:,ii]
        pred_slice = pred[:,:,ii]

        pred_slice[pred_slice<0] = 0 #bring the range to 0 and 1.
        pred_slice[pred_slice>1] = 1

        gt_slice_laplace = laplace(gt_slice)        
        pred_slice_laplace = laplace(pred_slice)

        hfn_slice = np.sum((gt_slice_laplace - pred_slice_laplace) ** 2) / np.sum(gt_slice_laplace **2)
        hfn_total.append(hfn_slice)

    return np.mean(hfn_total)

def get_target(fname):
   
   data = np.load(fname)
   num_slices =  data.shape[0]
#    print("num_slice",num_slices)
   kspace = data[20:num_slices-20,:,:]
#    print("shape",kspace.shape)
   kspace_cplx = kspace[:,:,:,0] + 1j*kspace[:,:,:,1]
   
   kspace = np.fft.fftshift(kspace_cplx)
   target = np.fft.ifft2(kspace_cplx)
   target_abs = np.abs(target)
   
   
   for i in range(20,num_slices-20):
        target_abs[i-20] = target_abs[i-20] / target_abs[i-20].max()
   
#    target_abs = target_abs/np.max(target_abs)
   
   return target_abs

def get_pred(recons_file):   # normalize the predictions in range 0 to 1 
    
    
        
        recons = np.load(recons_file)
        num_slices = recons.shape[0]
        for i in range(num_slices):
                recons[i] = recons[i] / recons[i].max()
                # print("reons_max",i,recons[i].max())   
        return recons
         
    
    
    


def mse(gt, pred):
    # print("gt,pred",gt.dtype,pred.dtype)
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return compare_psnr(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    #return compare_ssim(
    #    gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    #)
    return compare_ssim(gt,pred,multichannel=True, data_range=gt.max())
   

METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim
    # HFN=hfn
)


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        self.metrics = {
            metric: Statistics() for metric in metric_funcs
        }

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {
            metric: stat.mean() for metric, stat in self.metrics.items()
        }

    def stddevs(self):
        return {
            metric: stat.stddev() for metric, stat in self.metrics.items()
        }
# '''
#     def __repr__(self):
#         means = self.means()
#         stddevs = self.stddevs()
#         metric_names = sorted(list(means))
#         return ' '.join(
#             f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
#         )
# ''' 
    def get_report(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
        )




def evaluate(args, recons_key):
    metrics = Metrics(METRIC_FUNCS)
    print("predictions picked up from :",args.predictions_path)
    for tgt_file in tqdm(args.target_path.iterdir()):
        
        
        # print ("target",tgt_file)
        target = get_target(tgt_file)
        recons_file = args.predictions_path / tgt_file.name
        # print("recons",recons_file)
        
        # recons = np.load(recons_file)
        if args.normalized=='Normalized':
            print('Normalizing reconstructions.....')
            recons = get_pred(recons_file)
        else:
            recons = np.load(recons_file)

        
        # print(target.shape , recons.shape)
    #     with h5py.File(tgt_file) as target, h5py.File(
    #       args.predictions_path / tgt_file.name) as recons:
    #         target = target[recons_key].value
    #         recons = recons['reconstruction'].value
    #         recons = np.transpose(recons,[1,2,0])
    #         #print (target.shape,recons.shape)
        metrics.push(target, recons)
            
    return metrics


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target-path', type=pathlib.Path, required=True,
                        help='Path to the ground truth data')
    parser.add_argument('--predictions-path', type=pathlib.Path, required=True,
                        help='Path to reconstructions')
    parser.add_argument('--report-path', type=pathlib.Path, required=True,
                        help='Path to save metrics')
    parser.add_argument('--normalized', type=str, required=True,
                        help='whether to normaalize the reconstructions')
    

    args = parser.parse_args()

    recons_key = 'volfs'
    metrics = evaluate(args, recons_key)
    metrics_report = metrics.get_report()

    if args.normalized =='Normalized':
        with open(args.report_path/'normalized_report.txt','w') as f:
            f.write(metrics_report)
    else:
        with open(args.report_path/'unnormalized_report.txt','w') as f:
            f.write(metrics_report)

    print("check report @ :",args.report_path)
