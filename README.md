## Deep Convolutional Encoder-Decoder Networks for Dynamical Multi-Phase Flow Models
[Deep Convolutional Encoder-Decoder Networks for Uncertainty Quantification of Dynamic Multiphase Flow in Heterogeneous Media](https://arxiv.org/abs/1807.00882)

Shaoxing Mo, [Yinhaozhu](https://scholar.google.com/citations?user=SZmaVZMAAAAJ&hl=en&oi=sra), [Nicholas Zabaras](https://www.zabaras.com/), [Xiaoqing Shi](https://scholar.google.com/citations?user=MLKqgKoAAAAJ&hl=en&oi=sra), Jichun Wu

PyTorch implementation of deep convolutional nueral networks for dynamical multi-phase flow models with discontinuous outputs and subsequent uncertainty quantification. We treat time as an input to network to predict the time-dependent outputs of the dynamic system.

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
Illustration of the repo structure. The training data are obtained by reorganizing the original data (see Section 3.3 in [Mo et al. (2018)](https://arxiv.org/abs/1807.00882)) to characterize the system dynamics.

# Start with a Pre-trained Model
The pretrained models of networks [with the MSE loss](https://drive.google.com/file/d/1VtcpywvbUzTEXr1IU7GZtewXi1UWCuz2/view?usp=sharing) and [with the MSE-BCE loss](https://drive.google.com/open?id=1-CPrGxw6fnIeXFRr1sHhbnOZffGoyWT7) are available on Google Drive. One can plot the images provided using the script "post_usePretrainedModel.py".

# Network Training
```
python3 train_time.py
```
# Citation
See [Mo et al. (2018)](https://arxiv.org/abs/1807.00882) for more information. If you find this repo useful for your research, please consider to cite:
```
@ARTICLE{2018arXiv180700882M,
       author = {{Mo}, Shaoxing and {Zhu}, Yinhao and {Zabaras}, Nicholas and {Shi},
        Xiaoqing and {Wu}, Jichun},
        title = "{Deep convolutional encoder-decoder networks for uncertainty
        quantification of dynamic multiphase flow in heterogeneous media}",
      journal = {arXiv e-prints},
     keywords = {Statistics - Machine Learning, Computer Science - Machine Learning},
         year = 2018,
        month = Jul,
          eid = {arXiv:1807.00882},
        pages = {arXiv:1807.00882},
archivePrefix = {arXiv},
       eprint = {1807.00882},
 primaryClass = {stat.ML},
       adsurl = {https://ui.adsabs.harvard.edu/\#abs/2018arXiv180700882M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
or:
```
Mo, S., Zhu, Y., Zabaras, N., Shi, X., & Wu, J. (2018). Deep convolutional encoder-decoder networks for 
uncertainty quantification of dynamic multiphase flow in heterogeneous media. arXiv preprint arXiv:1807.00882.
```
Related article: [Zhu, Y., & Zabaras, N. (2018). Bayesian deep convolutional encoder–decoder networks for surrogate modeling and uncertainty quantification. J. Comput. Phys., 366, 415-447.](https://www.sciencedirect.com/science/article/pii/S0021999118302341)

# Questions
Contact Shaoxing Mo (smo@smail.nju.edu.cn) or Nicholas Zabaras (nzabaras@gmail.com) with questions or comments.
