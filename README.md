# Dual-Encoder-Unet
This repository provides reproducible code for our paper titled: "**Dual-Encoder-Unet for fast MRI Reconstruction**"
<br>
The paper got accepted in IEEE 17th International Symposium on Biomedical Imaging Workshops (ISBI Workshops-2020) held at Iowa City, USA.
<br>
The paper is added to IEEE Xplore and can be found [here](https://ieeexplore.ieee.org/document/9153453).
<br>

* Our work provides a deep learning based strategy by simultaneously optimizing both the raw kspace data and undersampled image data for reconstruction. 
* Our experiments demonstrate that, such a hybrid approach can potentially improve reconstruction, compared to deep learning networks that operate solely on a single domain.

<img src="brain/images/dualencoder_fin.png">

* Brain
    * dAutomap
    * U-Net
    * Dual-Encoder Unet
* Cardiac
    * dAutomap
    * U-Net
    * Dual-Encoder Unet

## Experients with Brain MRI

### dataset
The Brain MRI data is collected from [Calgary-Campinas Public Brain MR Datasest](https://sites.google.com/view/calgary-campinas-dataset/home). 

### results
<img src="brain/brain.png">

## Experiments with Cardiac MRI

### dataset
The Cardiac MRI data is collected from [Automated Cardiac Diagnosis Challenge (ACDC)](https://acdc.creatis.insa-lyon.fr/description/databases.html). 

### results
<img src="cardiac/cardiac.png">


<!-- <img src="brain/images/fs4.png" width = 175>  <img src="brain/images/us4.png" width = 175>
<img src="brain/images/dauto4.png" width = 175>
<img src="brain/images/unet4.png" width = 175>
<img src="brain/images/dual4.png" width = 175>

<img src="cardiac/images/fs_cardiac.png" width = 175>  <img src="cardiac/images/us_cardiac.png" width = 175>
<img src="cardiac/images/dauto_cardiac.png" width = 175>
<img src="cardiac/images/unet_cardiac.png" width = 175>
<img src="cardiac/images/dual_cardiac.png" width = 175> -->

## cite
if you use our work in your experiments please cite as :


```
    @inproceedings{jethi2020dual,
    title={Dual-Encoder-Unet For Fast Mri Reconstruction},
    author={Jethi, Amrit Kumar and Murugesan, Balamurali and Ram, Keerthi and Sivaprakasam, Mohanasankar},
    booktitle={2020 IEEE 17th International Symposium on Biomedical Imaging Workshops (ISBI Workshops)},
    pages={1--4},
    year={2020},
    organization={IEEE}
    }
```
## questions
if you face any problems or want to discuss ideas plz reach out to me at: ee18s037@smail.iitm.ac.in
