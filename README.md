# My Experiments with Convolutional and Variational Autoencoders.

<center>
  <img src="/images/autoencoder.png" align="middle" width="90%" height= "60%" alt= "autoencoders"/>
</center>

## What are autoencoders?
- To put in simple terms, Autoencoders are simply feature extractors with additional task of\
 reconstructiong the given input which maybe images or videos with minimum losses.
- In Modern neural networks the autoencoders are utilized for many other purposes like that of \
 image segmentation (refer U-nets), image denoising, image completion or also in modern reinforcement learning algorithms (Graph neural networks).
 
## Here is my experience of trying to use convolutional autoencoders.
- Convolutional Autoencoders work well untill scaled down to 1x1 feature levels.
- One can not be hopeful of generating a new image using only the trained conv-generator.
- Making the autoencoder conditional does not help to improve the quality of images reproduced.
 

This is my best attempt at reproducing the images using a conv-autoencoder for code size of 2x2 features.
Dataset         - STL10
Loss Criterion  - MSE Loss(0.0028)
Network Type    - Simple Convolutional Network

<p>
  <img src="Feature_size_2x2/train_epoch_993_stl.png" align="middle" width="90%" alt= "Reconstructions"/>
</p> 

## What are Variational Autoencoders.
- Variational Autoencoders are a link between the autoencoders and generative networks. Simply, they are \
 also able to sample and create new images better than conv-autoencoders.
- It has a additional KL-Divergence loss which aims to remove inter-dependibility of each feature of the 
 bottleneck part and spread it in a lower bound known to resample it.

<center>
  <img src="images/vae_model.png" align="middle" width="90%" height= "60%" alt= "VAE"/>
</center>


## Here is my experience of trying to use variational autoencoders.
- Here I wanted to use only 40 features so Image reconstructions are not very clear but should be better with \
 more features on last part of encoder
- I have also made only the decoder part of network conditional by supplying additional labels. (previous experiments \
 of making the whole network conditional also gives same results).
 
Dataset         - Cifar10
Loss Criterion  - Kl+MSE loss (~13000) 
Feature size    - 40
Network Type    - VAE
 
<p>
  <img src="vae/train_epoch_99.png" align="middle" width="90%" alt= "Reconstructions"/>
</p> 

### Samples reproduced only using the trained generator


<p>
  <img src="vae/test_epoch_99.png" align="middle" width="90%" alt= "Reconstructions"/>
</p>

I know it is not the best but it created this only using 40 features and also for a complex dataset then MNIST. 


### TO Do (Future work)
- Image completion.
- Denoising images.