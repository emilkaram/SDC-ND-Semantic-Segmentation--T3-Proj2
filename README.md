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
    
    
### Define the convolution layers and skip connections:
    def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    
    # 1x1 conv @ layer 7
    layer7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes ,1 , padding='same' , 
                                kernel_initializer= tf.random_normal_initializer(stddev = 0.01),
                                kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    # upsample layer 4 pooling
    layer4_up = tf.layers.conv2d_transpose(layer7_1x1 , num_classes , 4 , strides=(2,2) , padding='same',
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    
    # 1x1 conv @ layer 4
    layer4_1x1 = tf.layers.conv2d(vgg_layer4_out , num_classes , 1 , padding='same',
                                 kernel_initializer=tf.random_normal_initializer(stddev =0.01)
                                 ,kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # skip connection
    layer4_out_all = tf.add(layer4_up , layer4_1x1)
    
    
    
     # upsample layer 3 pooling
    layer3_up = tf.layers.conv2d_transpose(layer4_out_all , num_classes , 4 , strides=(2,2) , padding='same',
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    
    # 1x1 conv @ layer 3
    layer3_1x1 = tf.layers.conv2d(vgg_layer3_out , num_classes , 1 , padding='same',
                                  kernel_initializer=tf.random_normal_initializer(stddev =0.01)
                                 ,kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # skip connection
    layer3_out_all = tf.add(layer3_up , layer3_1x1)
 

    # upsample final_layer
    nn_last_layer = tf.layers.conv2d_transpose(layer3_out_all , num_classes , 16 , strides=(8,8), padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                            kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    


    return nn_last_layer
    

### Define the loss fuction and optimizer:
    def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    
    # biuld and reshape the logits with 2D tensor (row = pixel , col = class) 
    logits = tf.reshape(nn_last_layer , (-1 , num_classes))
    correct_label = tf.reshape(correct_label,(-1,num_classes))
    
    #loss fucntion
    cross_entrop_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits , labels = correct_label))
    
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(cross_entrop_loss)
        
    
    return logits, train_op, cross_entrop_loss
    
 
 ### train the model:
 
    def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    sess.run(tf.global_variables_initializer())
    print('training started....\n')
    
    for epoch in range(epochs):
        print ("EPOCH {} ".format(epoch+1))
        for image , label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op , cross_entropy_loss], 
                                     feed_dict={input_image:image , correct_label:label , 
                                                keep_prob:0.5 , learning_rate:.0009})
            
            print ("Loss: = {:.3f}".format(loss))
        print()
        
        
        
### run the model:
    def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        batch_size = 5
        epochs = 50
        
        # define tensorflow placehoders
        correct_label = tf.placeholder(tf.int32 ,[None,None,None,num_classes] , name ='correct_label')
        learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
        
        input_image , keep_prop , vgg_layer3_out , vgg_layer4_out , vgg_layer7_out =load_vgg(sess , vgg_path)
        
        nn_last_layer = layers(vgg_layer3_out , vgg_layer4_out , vgg_layer7_out,num_classes)
        
        logits , train_op , cross_entropy_loss = optimize(nn_last_layer , correct_label , learning_rate , num_classes)
        
  
  ### train nn
       # TODO: Train NN using the train_nn function
        train_nn(sess , epochs , batch_size , get_batches_fn, train_op , cross_entropy_loss , input_image , correct_label,
                keep_prop , learning_rate)
     
 ### Save the infrences samples
       # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prop, input_image)
        print('saved')
        
                
                
        
        
    
      



### Run
Run the following command to run the project:
python main.py

### conculion
I trained the model with 6 , 20 , 50 epochs with patch size =5 , learning rate =  0.0009
I added layers.l2_regularizer(1e-3) to all layers for better performance.
I used layers 3, 4 and 7 of VGG16 to create skip layers for a fully convolutional network based on paper Fully [Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1605.06211.pdf)

![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/images/img_7.png)


### The loss improved after 10 epoch
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/loss/loss_50_epoch.png)


### inference_samples @ 6 epochs:
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/6_epochs/um_000047.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/6_epochs/um_000087.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/6_epochs/um_000088.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/6_epochs/umm_000068.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/6_epochs/uu_000059.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/6_epochs/uu_000097.png)


### inference_samples @ 20 epochs
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/20_epoch/um_000047.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/20_epoch/um_000087.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/20_epoch/um_000088.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/20_epoch/umm_000068.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/20_epoch/uu_000032.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/20_epoch/uu_000053.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/20_epoch/uu_000059.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/20_epoch/uu_000071.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/20_epoch/uu_000097.png)


### inference_samples @ 50 epochs

![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/50_epoch/um_000044.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/50_epoch/um_000046.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/50_epoch/um_000059.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/50_epoch/um_000065.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/50_epoch/um_000087.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/50_epoch/um_000088.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/50_epoch/umm_000068.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/50_epoch/umm_000083.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/50_epoch/umm_000086.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/50_epoch/uu_000013.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/50_epoch/uu_000022.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/50_epoch/uu_000042.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/50_epoch/uu_000059.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/50_epoch/uu_000071.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/50_epoch/uu_000078.png)
![](https://github.com/emilkaram/SDC-ND-Semantic-Segmentation-TensorFlow-T3-Proj2/blob/master/runs/50_epoch/uu_000086.png)


