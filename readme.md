# Differentiable Morphing

### Image morphing without reference points by applying warp maps and optimizing over them.  
Differentiable Morphing is machine learning algorithm that can morph any two images without reference points. It called "differentiable morphing" because neural network here is not used in traditional data to label mapping sense, but as an easy way to solve optimization problem where one image is mapped to another via warp maps that are found by gradient descent. So after maps are found there is no need for the network itself.

## Results
![example 1](images/example_1.gif)
![example 2](images/example_2.gif)
![example 3](images/example_3.gif)

## Dependencies

Tensorflow 2.1.3 and above.

## Example notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chigozienri/DiffMorph/blob/master/DiffMorph.ipynb)

## Usage

Install proper dependencies:

```
pip install -r requirements.txt
```

Use the program:

```bash
morph.py -s images/img_1.jpg -t images/img_2.jpg
```
-s Source file  
-t Target file  

Unnecessary parameters:  
-e Number of epochs to train maps on training stage  
-a Addition map multiplier  
-m Multiplication map multiplier  
-w Warp map multiplier  
-add_first If true add map would be applied to the source image before mult map. (might work better in some cases)

## Idea

Suppose we want to produce one image from another in a way that we use as much useful information as possible, so if two given images share any similarities between them we make use of these similarities. 

![toy_example](images/toy_example.jpg)  

After several trials I found out that the best way to achieve such effect is to use following formula.  

![formula](images/formula.jpg)  

Here "Mult map" removes unnecessary parts of an image and shifts color balance, "Add map" creates new colors that are not present in original image and "Warp map" distort an image in some way to reproduce shifting, rotation and scaling of objects. W operation is [dense_image_warp](https://www.tensorflow.org/addons/api_docs/python/tfa/image/dense_image_warp) method that present in tensorflow and usually used for optical flow estimation tasks. 

All maps are found by gradient descent using very simple convolution network. Now, by applying alpha scaling parameter to every map we will get smooth transition from one image to another without any loss of useful data (at least for the given toy example).  

![transition](images/transition.jpg) 


## Thoughts

Notice that all maps produced generate somewhat meaningful interpolation without any understanding of what exactly present in the images. That means that warp operation might be very useful in images processing tasks. In some sense warp operation might be thought as long range convolution, because it can "grab" data from any point of an image and reshape it in some useful way. Therefore it might be beneficial to use warp operation in classification tasks and might allow networks be less susceptible to small perturbations of the data. But especially, it should be beneficial to use in generation task. It should be much easier to produce new data by combining and perturbating several examples of known data points than to learn a function that represents all data points at ones.
