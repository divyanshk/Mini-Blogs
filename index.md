---
layout: default
---

[Deconvolution Layer](#deconv)  
[Batch Normalization](#batchnorm)  
[SqueezeNet ](#squeezenet)  

---

## <a name="deconv"></a>Deconvolution Layer

* torch.nn.ConvTranspose2d in PyTorch
* ambiguous name, no deconvolutions
* a deconvolution layer maps from a lower to higher dimension, a sort of upsampling
* the transpose of a non-padded convolution is equivalent to convolving a zero-padded input
* zeroes are inserted between inputs which cause the kernel to move slower, hence also called fractionally strided convolution
* deconv layers allow the model to use every point in the small image to “paint” a square in the larger one
* deconv layers have uneven overlap in the output, conv layers have overlap in the input
* leads to the problem of checkerboard artifacts
* resize-convolution instead transposed-convolution to avoid checkerboard artifacts

References
* [Convolution Arithmatic](http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html){:target="_blank"}
* [Distil Blog Post](https://distill.pub/2016/deconv-checkerboard/){:target="_blank"}
* [Original Paper](http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf){:target="_blank"}

---

## <a name="batchnorm"></a>Batch Normalization

* torch.nn.BatchNorm2d in PyTorch
* normalizses the data in each batch to have zero mean and unit covariance
* provides some consistency between layers by reducing internal covariate shift
* allows a higher learning rate to be used, reduces the learning time
* after normalizing the input, it is squased through a linear function with parameters gamma and beta
* output of batchnorm = gamma * normalized_input + beta
* having gamma and beta allows the network to choose how much 'normalization' it wants for every feature; shift and scale

References
* [Andrej Karapathy's lecture](https://www.youtube.com/watch?v=gYpoJMlgyXA&feature=youtu.be&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&t=3078){:target="_blank"}
* [Original Paper](https://arxiv.org/abs/1502.03167){:target="_blank"}
* [Read this later](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html){:target="_blank"}

---

## <a name="squeezenet"></a>SqueezeNet

* A conv net aimed at drastically reducing the number of parameter without much loss to accuracy
* Uses 1x1 filters instead of 3x3
* 1x1 filters work by capturing information on the channels of the pixels as compared to the neighborhood
* Uses a 'fire' module, composed of 'squeeze' and 'expand' layers
* The number of filters in the squeeze layer is restricted, thereby reducing the input channels to the expand layer
* Downsampling late into the network to have larger activation maps
* No fc layers

References
* [Original paper](https://arxiv.org/pdf/1602.07360.pdf){:target="_blank"}
* [KDNuggest summary](https://www.kdnuggets.com/2016/09/deep-learning-reading-group-squeezenet.html){:target="_blank"}

---