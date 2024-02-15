# Image colorizer

Here is a project I have been working on to solve the task of image colorization
using deep learning. 

![Colorization Output](https://github.com/Ricardicus/colorizer-gan/blob/master/outputs/23__dim_256__adv_0p5__compl_72/outputs/collection_image____2.png "Colorization Project")

I trained on a data set of landscapes, taken from Kaggle ([Landscape Pictures](https://www.kaggle.com/datasets/arnaud58/landscape-pictures)) and I got results I was satisfied with.
In the image above, we have a grid of 6x3 images. The the left, the 3x3 grid of images to the left, we have gray scaled images taken from the internet that are not part of the
training set. And to the right, the 3x3 grid of images to the right, we have the corresponding output from 
a model I trained with the code in this repo.

## Adversarial training

Just training the generator network, the UNet model, on optimization for some kind of norm of the output will likely 
create a model that produces mono-colored images. The GAN effect produces resemblance with the training data in the output,
which for example can mean that the color intensity of the output will resemble that of images trained on.

# Technical report

I did this as part of a course at the university, so I wrote a technical report on it here: [Colorization Project PDF](https://github.com/Ricardicus/colorizer-gan/blob/master/report/Colorization_Project.pdf)



