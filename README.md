## Deep Convolutional Encoder-Decoder Networks for Dynamical Multi-Phase Flow Models
[Deep Convolutional Encoder-Decoder Networks for Uncertainty Quantification of Dynamic Multiphase Flow in Heterogeneous Media](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2018WR023528)

[Shaoxing Mo](https://scholar.google.com/citations?hl=en&user=G6ac1xUAAAAJ&view_op=list_works&gmla=AJsN-F4ses_YhFsF-w2sFZLhacR7vrVyN1272g_B7XQyGbYsvy_6ReJpe4ChndNy_cFQ7UqXCSi82UiLjMB2dKyqSj8x5DaPRg), [Yinhaozhu](https://scholar.google.com/citations?user=SZmaVZMAAAAJ&hl=en&oi=sra), [Nicholas Zabaras](https://www.zabaras.com/), [Xiaoqing Shi](https://scholar.google.com/citations?user=MLKqgKoAAAAJ&hl=en&oi=sra), Jichun Wu

PyTorch implementation of deep convolutional nueral networks for dynamical multi-phase flow models with discontinuous outputs and for subsequent uncertainty quantification. We treat time as an input to network to predict the time-dependent outputs of the dynamic system.

![alt text](https://github.com/njujinchun/dcedn-gcs/blob/master/images/N_1600_output_5_ls50_var1.png)
The first column is the forward model predictions for the pressure (left) and discontinuous saturation (right) fields at t=100, 150, and 200 days. The second and third columns are the network predictions and predicted errors, respectively.

# Two-Stage Network Training Combining Regression and Segmentation Losses
To improve the approximation accuracy for the irregular discontinuous saturation front, we binarize (0 or 1) the saturation field and the resulting image is added as an additional output channel to the network. A binary cross entropy (BCE) loss is used for the the two-class segmentation task ([CNN-MSE-BCE loss](https://github.com/njujinchun/dcedn-gcs/tree/master/CNN-MSE-BCE%20loss)). The network with a MSE loss ([CNN-MSE loss](https://github.com/njujinchun/dcedn-gcs/tree/master/CNN-MSE%20loss)) solely is also provided for comparison.
![alt text](https://github.com/njujinchun/dcedn-gcs/blob/master/images/Sg_binarized.png)
Left: Discontinuous saturation field. Right: The corrresponding binarized image.

# Network Architecture
![alt](https://github.com/njujinchun/dcedn-gcs/blob/master/images/DCEDN.png)
The network is fully convolutional without any fully-connnected layers and is an alternation of dense blocks and transition (encoding/decoding) layers.

# Dependencies
* python 3
* PyTorch 0.4
* h5py
* matplotlib
* seaborn

# Datasets, Pretrained Model, and Forward Model Input Files
The datasets used, pretrained models, input files for the forward model, and needed scripts have been uploaded to Google Drive and can be downloaded using this link [https://drive.google.com/drive/folders/1keg9HwP3bs9JUCyqYflKNwIHwep2CD6r?usp=sharing](https://drive.google.com/drive/folders/1keg9HwP3bs9JUCyqYflKNwIHwep2CD6r?usp=sharing)

# Repo Structure
![alt](https://github.com/njujinchun/dcedn-gcs/blob/master/images/Repo-structure.png)
Illustration of the repo structure. The training data are obtained by reorganizing the original data (see Section 3.3 in [Mo et al. (2018)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2018WR023528)) to characterize the system dynamics.

# Start with a Pre-trained Model
The pretrained models of networks [with the MSE loss](https://drive.google.com/file/d/1VtcpywvbUzTEXr1IU7GZtewXi1UWCuz2/view?usp=sharing) and [with the MSE-BCE loss](https://drive.google.com/open?id=1-CPrGxw6fnIeXFRr1sHhbnOZffGoyWT7) are available on Google Drive. One can plot the images provided using the script "post_usePretrainedModel.py".

# Network Training
```
python3 train_time.py
```
# Citation
See [Mo et al. (2018)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2018WR023528) for more information. If you find this repo useful for your research, please consider to cite:
```
@article{moetal2018,
author = {Mo, Shaoxing and Zhu, Yinhao and Zabaras, Nicholas, J and Shi, Xiaoqing and Wu, Jichun},
title = {Deep convolutional encoder-decoder networks for uncertainty quantification of dynamic multiphase flow in heterogeneous media},
journal = {Water Resources Research},
volume = {},
number = {},
pages = {},
keywords = {Multiphase flow, geological carbon storage, uncertainty quantification, deep neural networks, high-dimensionality, response discontinuity},
doi = {10.1029/2018WR023528},
url = {https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2018WR023528},
eprint = {https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2018WR023528},
}
```
or:
```
Mo, S., Y. Zhu, N.J Zabaras, X. Shi, and J., Wu. (2018), Deep convolutional encoder‐decoder networks for 
uncertainty quantification of dynamic multiphase flow in heterogeneous media, Water Resources Research, 
https://doi.org/10.1029/2018WR023528
```
Related article: [Zhu, Y., & Zabaras, N. (2018). Bayesian deep convolutional encoder–decoder networks for surrogate modeling and uncertainty quantification. J. Comput. Phys., 366, 415-447.](https://www.sciencedirect.com/science/article/pii/S0021999118302341)

# Questions
Contact Shaoxing Mo (smo@smail.nju.edu.cn) or Nicholas Zabaras (nzabaras@gmail.com) with questions or comments.
