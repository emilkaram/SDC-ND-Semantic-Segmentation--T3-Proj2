# SDC-ND-Semantic-Segmentation-T3-Proj2


![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/images/img_1.png)

### Introduction:
In this project, I labeled the pixels of a road in images using a Fully Convolutional Network (FCN)


### Semantic segmentation
Semantic segmentation is different from image classification. 
In image classification,the classifier will classify objects based on its labels (supervised learning) where as semantic segmentation algorithm will segment objects in an image pixlewise that means each pixel is assigned to a specific class in the image.


### Convolutional Neural Network(CNN) 
In CNN, input layer followed by convolution layer, then it is connected to fully connected layer followed softmax to classify the image. CNN is to classify if the image has got particular object, but it more difficult to answer “where is the object in the image”.
This is because of fully connected layer doesn’t preserve spatial information. 

![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/images/image3.jpeg)


### Fully Convolutional Network(FCN)
Encoder-decoder architecture is used where encoder reduces the spatial dimension with pooling layers and decoder upsample the object details and spatial dimension the interface layer between the Encoder and Decoder is 1x1 convolution layer. 
Skip connections trick from encoder to decoder to preserve spatial information.

![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/images/img_2.png)


### Setup
GPU: main.py will check to make sure you are using GPU
if you don't have a GPU on your system, you can use AWS or another cloud computing platform.

### Frameworks and Packages
Make sure you have the following is installed:

Python 3
TensorFlow
NumPy
SciPy

### Dataset
Download the[Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip). Extract the dataset in the data folder. This will create the folder data_road with all the training a test images.

## implemantation:
### Load Vgg
    def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess , [vgg_tag] ,vgg_path)
    image_input= tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    keep_prop  = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    
    return image_input, keep_prop, layer3_out, layer4_out, layer7_out
    
      



### Run
Run the following command to run the project:
python main.py


