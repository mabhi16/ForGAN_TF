# ForGAN_TF
TensorFlow-Keras implementation for ForGAN

This is a tensorflow-keras implementation for the Forecasting GAN (ForGAN) which was originally implemented by Koochali et. al (https://arxiv.org/pdf/1903.12549v1.pdf).

GAN was originally implemented to handle the univariate analysis and prediction, with very few modifications I have tried to do multi-variate analysis as well.

The Data used is the Stock data of GE company available in Kaggle. I have uploaded the the data sample as well.  

components.py makes compoments of the GANs such as discriminator, generator and also a combinational network which is required to train the generator.

utils,py contains crutial functions required for KLD calculation and data input.

GE_ForGAN.py is the initail implementation where it do univariate analysis and output single prediction.

mutlistep.py has some adaptation and it can be used to do multi-varaiate analysis and predection as well.

I have also included few of the graph showing the prediction of the network.
