# practical-deep-learning-for-computer-vision-with-python

### The architecture of CNNs is generally composed of 3 distinct types of layers:

    Convolutional Layers
    Pooling Layers
    Fully-Connected Layers

Generally speaking - convolutional layers are interlaced with pooling layers, and this symbiosis can be seen as one 
unit of a CNN, while the fully-connected layers can be seen as a separate unit on top of the first one. 
The convolutional (and pooling) layers are known to be feature extractors and extract the features that make a class 
what it is. The fully-connected layers are, essentially, an entire MLP on top of the feature extractors, 
and act as a classifier.


To that end - a CNN is really, a network of feature extractors and a classifier network on top of it.

## The Convolutional Layer

`Convolutional Layers perform feature extraction and result in feature maps. 
Feature extraction is achieved through convolution, and repetitive convolutions across an entire image result in a 
feature map.`

A feature map is (typically) a 3D tensor (width x height x depth). 

A convolution, in layman's terms, in the context of Deep Learning, is the multiplication of of two matrices. 
One matrix is a filter/kernel (a set of a neuron's weights) that we slide over the image, 
and the other matrix is the part of the image covered by the filter.

    model = keras.models.Sequential([
        # 64 filters, with a size of `3x3`
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=[150, 150, 3]),
        # You can also simply add one dimension for the size, it'll be symetrical `(3)`
        keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2), (2, 2)),
        keras.layers.BatchNormalization(),
    
        keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        # Same goes for MaxPooling `(2, 2)` is equal to `(2, 2), (2, 2)`
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.BatchNormalization(),
        
        keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
        keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.BatchNormalization(),
        
        keras.layers.Flatten(),    
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3, seed=2),
        keras.layers.Dense(6, activation='softmax')
    ])