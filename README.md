# Image colorizer

Here is a project I have been working on to solve the task of image colorization
using deep learning.

![Colorization Output](https://github.com/Ricardicus/colorizer-gan/blob/master/outputs/23__dim_256__adv_0p5__compl_72/outputs/collection_image____2.png "Colorization Project")

I trained on a data set of landscapes, taken from kaggle and I got results I was satisfied with.

## Adversarial training

Just training the generator network, the UNet model, on optimization for some kind of norm of the output will likely 
create a model that produces mono-colored images. The GAN effect produces resemblance with the training data in the output,
which for example can mean that the color intensity of the output will resemble that of images trained on.

# Documentation/report

I will try to write a comprehensible report on what happened here. That is on my TODO list.

