# Image-Forgery-using-Deep-Learning
Image Forgery Detection using Deep Learning, implemented in PyTorch.


## Proposal
The whole framework: An RGB image, firstly, is divided into overlapping patches (64x64). Then, RGB patches are converted to the YCrCb color channel, before being scored by a network. Lastly, a post-processing stage is designed to refine predictions of the network and make a final conclusion on the authentication of the image.

<p align="center">
  <img src="pic/framework.png" width="800" alt="accessibility text">
</p>

The deep neural network is adapted from MobileNet-V2. However, we modify the original MobileNet-V2 to be more relevant to our problem. The picture below depicts the architecture modification.

<p align="center">
  <img src="https://github.com/AntiAegis/Image-Forgery-using-Deep-Learning/blob/master/pic/modification.png" width="400" alt="accessibility text">
</p>


## Experimental results
We have conducted a comprehensive evaluation on model configurations to show which factor improves the final
performance of the model. To figure out this, we define six configurations accompanied with the MobileNetV2, denoted
as MBN2, as the core. There are two color channels to be considered, namely RGB and YCrCb. Besides, three MobileNetV2 architectures are taken into account for comparing. The first architecture is MobileNetV2 trained from scratch, the second one is MobileNetV2 initialized with pre-trained weights from ImageNet, and the last one is modified MobileNetV2 trained from scratch.

<p align="center">
  <img src="https://github.com/AntiAegis/Image-Forgery-using-Deep-Learning/blob/master/pic/performance.png" width="400" alt="accessibility text">
  <img src="https://github.com/AntiAegis/Image-Forgery-using-Deep-Learning/blob/master/pic/computation.png" width="370" alt="accessibility text">
</p>


## Citation
If you find this work useful, please cite:
```
@article{
  title={Preserving Spatial Information to Enhance Performance of Image Forgery Classification},
  author={Hanh Phan-Xuan, Thuong Le-Tien, Thuy Nguyen-Chinh, Thien Do-Tieu, Qui Nguyen-Van, and Tuan Nguyen-Thanh},
  journal={International Conference on Advanced Technologies for Communications (ATC)},
  year={2019}
}
```

## Contact
* [Nguyen Chinh Thuy](https://github.com/thuyngch)
* [Do Tieu Thien](https://github.com/dotieuthien)
* [Nguyen Van Qui](https://github.com/nvqui97)
