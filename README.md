# Deep Convolutional Dncoder-Decoder Networks for Dynamical Multi-Phase Flow Models
[Deep Convolutional Encoder-Decoder Networks for Uncertainty Quantification of Dynamic Multiphase Flow in Heterogeneous Media](https://arxiv.org/abs/1807.00882)

Shaoxing Mo, [Yinhaozhu](https://scholar.google.com/citations?user=SZmaVZMAAAAJ&hl=en&oi=sra), [Nicholas Zabaras](https://www.zabaras.com/), [Xiaoqing Shi](https://scholar.google.com/citations?user=MLKqgKoAAAAJ&hl=en&oi=sra), Jichun Wu

PyTorch implementation of deep convolutional nueral networks for dynamical multi-phase flow models with discontinuous outputs and subsequent uncertainty quantification.

![alt text](https://github.com/njujinchun/dcedn-gcs/blob/master/images/N_1600_output_5_ls50_var1.png)
The first column is the forward model predictions for the pressure (left) and discontinuous saturation (right) fields at t=100, 150, and 200 days. The second and third columns are the network predictions and predicted errors, respectively.


To improve the approximation accuracy for the irregular discontinuous saturation front, we binarize (0 or 1) the saturation field and the resulting image is added as an additional output channel to the network. An binary cross entropy (BCE) loss is used for the the two-class segmentation task.
![alt text](https://github.com/njujinchun/dcedn-gcs/blob/master/images/Sg_binarized.png)
Left: Discontinuous saturation field. Right: The corrresponding binarized image.
