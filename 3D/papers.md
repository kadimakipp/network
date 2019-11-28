# 3D sensed Papers

##RGB-D
### 1.Convolutional-recursive deep learning for 3d object classification   
**Abstract**     `2012 NIPS`       
>Recent advances in 3D sensing technologies make it possible to easily record color
and depth images which together can improve object recognition. Most current
methods rely on very well-designed features for this new 3D modality. We introduce a model based on a combination of convolutional and recursive neural
networks (CNN and RNN) for learning features and classifying RGB-D images.
The CNN layer learns low-level translationally invariant features which are then
given as inputs to multiple, fixed-tree RNNs in order to compose higher order features. RNNs can be seen as combining convolution and pooling into one efficient,
hierarchical operation. Our main result is that even RNNs with random weights
compose powerful features. Our model obtains state of the art performance on a
standard RGB-D object dataset while being more accurate and faster during training and testing than comparable architectures such as two-layer CNNs.

One of the earlier methods for classifying RGB-D is a combination of a CNN and RNN.
The approach works by first learning low level translation invariant features with CNNs for both an RGB and
corresponding depth image independently. The learnt features are then passed into multiple, fixed tree RNNs to generate
more high level global features. RNNs were used here as they were seen to reduce convolutional and pooling functions
into a single efficient hierarchical operation, resulting in improvements with regard to computation time

### 2.Multimodal Deep Learning for Robust RGB-D Object Recognition  
**Abstract** `2015 IEEE`


### 3.Indoor Semantic Segmentation using depth information  
 `2013` `arXiv:1301.3572`
 
### 4.Learning Hierarchical Features for Scene Labeling
 `2013`

### 5.Learning rich features from RGB-D images for object detection and segmentation        
 `2014`
 
### 6.Structured Forests for Fast Edge Detection
 `2013`
 
### 7.Perceptual Organization and Recognition of Indoor Scenes from RGB-D Images    
  `2013`
 
### 64