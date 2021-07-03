import torch
from torch import nn
from torch.nn import functional as F
import os 
import numpy as np
import itertools
from data.transforms import ifft2,fft2,fftshift


def init_noise_(tensor, init):
    with torch.no_grad():
        return getattr(torch.nn.init, init)(tensor) if init else tensor.zero_()


def init_fourier_(tensor, norm='ortho'):
    """Initialise convolution weight with Inverse Fourier Transform"""
    with torch.no_grad():
        # tensor should have shape: (nc_out, nc_in, kx, ky)=(2*N, 2, N, kernel_size)
        nc_out, nc_in, N, kernel_size = tensor.shape

        for k in range(N):
            for n in range(N):
                tensor.data[k, 0, n, kernel_size // 2] = np.cos(2 * np.pi * n * k / N)
                tensor.data[k, 1, n, kernel_size // 2] = -np.sin(2 * np.pi * n * k / N)
                tensor.data[k + N, 0, n, kernel_size // 2] = np.sin(2 * np.pi * n * k / N)
                tensor.data[k + N, 1, n, kernel_size // 2] = np.cos(2 * np.pi * n * k / N)

        if norm == 'ortho':
            tensor.data[...] = tensor.data[...] / np.sqrt(N)

        return tensor

def get_refinement_block(model='automap_scae', in_channel=1, out_channel=1):
    if model == 'automap_scae':
        return nn.Sequential(nn.Conv2d(in_channel, 64, 5, 1, 2), nn.ReLU(True),
                             nn.Conv2d(64, 64, 5, 1, 2), nn.ReLU(True),
                             nn.ConvTranspose2d(64, out_channel, 7, 1, 3))
    elif model == 'simple':
        return nn.Sequential(nn.Conv2d(in_channel, 64, 3, 1, 1), nn.ReLU(True),
                             nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                             nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                             nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                             nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                             nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                             
                             nn.Conv2d(64, out_channel, 3, 1, 1))
    else:
        raise NotImplementedError


class GeneralisedIFT2Layer(nn.Module):

    def __init__(self, nrow, ncol,
                 nch_in, nch_int=None, nch_out=None,
                 kernel_size=1, nl=None,
                 init_fourier=True, init=None, bias=False, batch_norm=False,
                 share_tfxs=False, learnable=True,shift=False):
        """Generalised domain transform layer

        The layer can be initialised as Fourier transform if nch_in == nch_int
        == nch_out == 2 and if init_fourier == True.

        It can also be initialised
        as Fourier transform plus noise by setting init_fourier == True and
        init == 'kaiming', for example.

        If nonlinearity nl is used, it is recommended to set bias = True

        One can use this layer as 2D Fourier transform by setting nch_in == nch_int
        == nch_out == 2 and learnable == False


        Parameters
        ----------
        nrow: int - the number of columns of input

        ncol: int - the number of rows of input

        nch_in: int - the number of input channels. One can put real & complex
        here, or put temporal coil channels, temporal frames, multiple
        z-slices, etc..

        nch_int: int - the number of intermediate channel after the transformation
        has been applied for each row. By default, this is the same as the input channel

        nch_out: int - the number of output channels. By default, this is the same as the input channel

        kernel_size: int - kernel size for second axis of 1d transforms

        init_fourier: bool - initialise generalised kernel with inverse fourier transform

        init_noise: str - initialise generalised kernel with standard initialisation. Option: ['kaiming', 'normal']

        nl: ('tanh', 'sigmoid', 'relu', 'lrelu') - add nonlinearity between two transformations. Currently only supports tanh

        bias: bool - add bias for each kernels

        share_tfxs: bool - whether to share two transformations

        learnable: bool

        """
        super(GeneralisedIFT2Layer, self).__init__()
        self.nrow = nrow
        self.ncol = ncol
        self.nch_in = nch_in
        self.nch_int = nch_int
        self.nch_out = nch_out
        self.kernel_size = kernel_size
        self.init_fourier = init_fourier
        self.init = init
        self.nl = nl

        if not self.nch_int:
            self.nch_int = self.nch_in

        if not self.nch_out:
            self.nch_out = self.nch_in

        # Initialise 1D kernels
        idft1 = torch.nn.Conv2d(self.nch_in, self.nch_int * self.nrow, (self.nrow, kernel_size),
                                padding=(0, kernel_size // 2), bias=bias)
        idft2 = torch.nn.Conv2d(self.nch_int, self.nch_out * self.ncol, (self.ncol, kernel_size),
                                padding=(0, kernel_size // 2), bias=bias)

        # initialise kernels
        init_noise_(idft1.weight, self.init)
        init_noise_(idft2.weight, self.init)

        if self.init_fourier:
            if not (self.nch_in == self.nch_int == self.nch_out == 2):
                raise ValueError

            if self.init:
                # scale the random weights to make it compatible with FFT basis
                idft1.weight.data = F.normalize(idft1.weight.data, dim=2)
                idft2.weight.data = F.normalize(idft2.weight.data, dim=2)

            init_fourier_(idft1.weight)
            init_fourier_(idft2.weight)

        self.idft1 = idft1
        self.idft2 = idft2

        # Allow sharing weights between two transforms if the input size are the same.
        if share_tfxs and nrow == ncol:
            self.idft2 = self.idft1

        self.learnable = learnable
        self.set_learnable(self.learnable)
        self.shift = shift

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn1 = torch.nn.BatchNorm2d(self.nch_int)
            self.bn2 = torch.nn.BatchNorm2d(self.nch_out)

    def forward(self, X):
        # input shape should be (batch_size, nc, nx, ny)
        batch_size = len(X)
        if self.shift:
            X = fftshift(X,dim=(-2,-1))
        # first transform
        x_t = self.idft1(X)

        # reshape & transform
        x_t = x_t.reshape([batch_size, self.nch_int, self.nrow, self.ncol]).permute(0, 1, 3, 2)

        if self.batch_norm:
            x_t = self.bn1(x_t.contiguous())

        if self.nl:
            if self.nl == 'tanh':
                x_t = F.tanh(x_t)
            elif self.nl == 'relu':
                x_t = F.relu(x_t)
            elif self.nl == 'sigmoid':
                x_t = F.sigmoid(x_t)
            else:
                raise ValueError

        # second transform
        x_t = self.idft2(x_t)
        x_t = x_t.reshape([batch_size, self.nch_out, self.ncol, self.nrow]).permute(0, 1, 3, 2)

        if self.batch_norm:
            x_t = self.bn2(x_t.contiguous())

        if (self.shift):
            # return x_t
            return fftshift(x_t,dim=(-2,-1))
        else:
            return x_t

    def set_learnable(self, flag=True):
        self.learnable = flag
        self.idft1.weight.requires_grad = flag
        self.idft2.weight.requires_grad = flag


class dAUTOMAP(nn.Module):
    """
    Pytorch implementation of dAUTOMAP

    Decomposes the automap kernel into 2 Generalised "1D" transforms to make it scalable.
    """
    def __init__(self, input_shape, output_shape, tfx_params, tfx_params2=None):
        super(dAUTOMAP, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        if tfx_params2 is None:
            tfx_params2 = tfx_params

        self.domain_transform = GeneralisedIFT2Layer(**tfx_params)
        self.domain_transform2 = GeneralisedIFT2Layer(**tfx_params2)
        self.refinement_block = get_refinement_block('automap_scae', input_shape[0], output_shape[0])    #automap_scae

    def forward(self, x):
        """Assumes input to be (batch_size, 2, nrow, ncol)"""
        #
        # x = x.permute(0,3,1,2)
        # print("shape",x.shape)
        x_mapped = self.domain_transform(x)
        x_mapped = F.tanh(x_mapped)
        x_mapped2 = self.domain_transform2(x_mapped)
        x_mapped2 = F.tanh(x_mapped2)
        out = self.refinement_block(x_mapped)
        
        return out



'''
# cnn blocks used instead of unet

def lrelu():
    return nn.LeakyReLU(0.01, inplace=True)

def relu():
    return nn.ReLU(inplace=True)

def conv_block(n_ch, nd, nf=32, ks=3, dilation=1, bn=False, nl='lrelu', conv_dim=2, n_out=None):

    # convolution dimension (2D or 3D)
    if conv_dim == 2:
        conv = nn.Conv2d
    else:
        conv = nn.Conv3d

    # output dim: If None, it is assumed to be the same as n_ch
    if not n_out:
        n_out = n_ch

    # dilated convolution
    pad_conv = 1
    if dilation > 1:
        # in = floor(in + 2*pad - dilation * (ks-1) - 1)/stride + 1)
        # pad = dilation
        pad_dilconv = dilation
    else:
        pad_dilconv = pad_conv

    def conv_i():
        return conv(nf,   nf, ks, stride=1, padding=pad_dilconv, dilation=dilation, bias=True)

    conv_1 = conv(n_ch, nf, ks, stride=1, padding=pad_conv, bias=True)
    conv_n = conv(nf, n_out, ks, stride=1, padding=pad_conv, bias=True)

    # relu
    nll = relu if nl == 'relu' else lrelu

    layers = [conv_1, nll()]
    for i in range(nd-2):
        if bn:
            layers.append(nn.BatchNorm2d(nf))
        layers += [conv_i(), nll()]

    layers += [conv_n]

    return nn.Sequential(*layers)
    '''


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


class UnetModel(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, drop_prob)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, drop_prob)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input

        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            downsample_layer = stack.pop()
            layer_size = (downsample_layer.shape[-2], downsample_layer.shape[-1])
            output = F.interpolate(output, size=layer_size, mode='bilinear', align_corners=False)
            output = torch.cat([output, downsample_layer], dim=1)
            output = layer(output)
        return self.conv2(output)


class UnetModelParallelEncoder(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers1 = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        self.down_sample_layers2 = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])

        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers1 += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2

        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers2 += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2

        self.conv = ConvBlock(ch*2, ch, drop_prob)

        
        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 3, ch // 2, drop_prob)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 3, ch, drop_prob)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )

    def forward(self, input1,input2):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack1 = []
        stack2 = []
        output1 = input1
        output2 = input2

        # Apply down-sampling layers
        for layer in self.down_sample_layers1:
            # print("before downsample1",output1.shape)
            output1 = layer(output1)

            stack1.append(output1)
            output1 = F.max_pool2d(output1, kernel_size=2)
            # print ("down_path1",output1.shape)

        for layer in self.down_sample_layers2:
            # print("before downsample2",output2.shape)
            output2 = layer(output2)
            stack2.append(output2)
            output2 = F.max_pool2d(output2, kernel_size=2)
            # print ("down_path2",output2.shape)
        # print("end of downsampling",output1.shape,output2.shape)
        output = torch.cat([output1,output2],dim=1)
        # print ("after cat",output.shape)
        output = self.conv(output)
        # print("after_cat_conv",output.shape)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack1.pop(),stack2.pop()], dim=1)
            # print ("up1",output.shape)
            output = layer(output)
            # print("up2",output.shape)
        return self.conv2(output)

class dAUTOMAPDualEncoderUnet(nn.Module):
    def __init__(self,model1,model2):
        super(dAUTOMAPDualEncoderUnet,self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self,input_kspace,input):

        
        
        dautomap_pred = self.model1(input_kspace)
        
        # print("dautomap_pred",dautomap_pred.shape)
        # dautomap_pred = F.pad(dautomap_pred,(5,5,5,5),"constant",0)

        output = self.model2(dautomap_pred,input)

        return output,dautomap_pred


