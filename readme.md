# Differentiable Morphing

### Image morphing without reference points by applying warp maps and optimizing over them.  
It called "differentiable morphing" because neuronal network here is not used in traditional data to label mapping sence, but as an easy way to solve optimisation problem where one image is mapped to another via warp maps that are found by gradient dicent. So after maps are found there is no need for the network itself.

## Results
![example 1](images/example_1.gif)
![example 2](images/example_2.gif)
![example 3](images/example_3.gif)

## Dependencies

Tensorflow 2.4.0

## Usage

```bash
morph.py -s images/img_1.jpg -t images/img_2.jpg [-e 1000 -a 0.8 -m 0.8 -w 0.3]
```
-s Source file  
-t Target file  
  
Unnecessery parameters:  
-e Number of epochs to train maps on traning stage  
-a Addition map multiplyer  
-m Multiplication map multiplyer  
-w Warp map multiplyer  
  
## Idea
Suppose we want to produce one image from another in a way that we use as much useful information as possible, so if two given images share any similarities between them we make use of these similarities.

First thing we could do is to add some color data to the image in pixel space and see what we get. (In all examples operations maps have intentionally lower resolution than the image itself, so all nuances of operations would be more noticeable)  



As a last step we will add warping operation that already exist in tensorflow as [dense_image_warp](https://www.tensorflow.org/addons/api_docs/python/tfa/image/dense_image_warp) and usually used for optical flow estimation tasks. 

Now, by applying alpha scaling parameter to every map we will get smooth transition from one image to another without any loss of useful data (at least for the given toy example).

## Thoughts
Image warp operation might be thought as long range convolution, because it can "grab" data from any point of an image and reshape it in some useful way. Therefore it might be benefitial to use warp operation in calssification task. But especially, it should be benefitial to use in generation task. It should be much easier to produce new data by combining and perturbating several examples of known data points than to learn a function that represents all data points at ones.
