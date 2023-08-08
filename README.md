# practical-deep-learning-for-computer-vision-with-python
https://stackabuse.com/

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


For classifiers that work with a large number of classes and which have to make nuanced decisions - scoring by accuracy is simply unfair. 
I wouldn't say that I don't know what a street or a building are - but the 75% accuracy I got from the images earlier would imply a shaky 
knowledge of what these are! Accuracy as a metric, is essentially, Top-1 Categorical Accuracy and it's pretty difficult to get that score very high for 
complex problems. Most of the papers you'll read on complex image classification will have some level of Top-K accuracy besides Top-1 accuracy, 
depending on the number of classes in the dataset. If it's 1000 classes, like with ImageNet - people commonly also calculate a Top-5 accuracy score 
besides the Top-1 accuracy score.

    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy',
                           keras.metrics.TopKCategoricalAccuracy(k=2)])

model.summary()

    `Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 150, 150, 64)      1792      
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 150, 150, 64)      36928     
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 75, 75, 64)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 75, 75, 128)       73856     
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 75, 75, 128)       147584    
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 37, 37, 128)       0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 37, 37, 256)       295168    
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 37, 37, 256)       590080    
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 18, 18, 256)       0         
    _________________________________________________________________
    flatten (Flatten)            (None, 82944)             0         
    _________________________________________________________________
    dense (Dense)                (None, 64)                5308480   
    _________________________________________________________________
    batch_normalization (BatchNo (None, 64)                256       
    _________________________________________________________________
    dropout (Dropout)            (None, 64)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 6)                 390       
    =================================================================
    Total params: 6,454,534
    Trainable params: 6,454,406
    Non-trainable params: 128`

Learning Rate 

So - how do you figure out which learning rate to use? Note that this isn't a question of how you should change the learning rate during training. 
This refers to the initial learning rate. In the beginning, you want to maximize the change in loss (with a slope pointing to the minimum). 
This tends to be a higher learning rate, though, you can go too high even here, and make it difficult to reduce it to a more sustainable number. 
Let's recap, once you start, there are various methods of updating the learning rate during training to optimize 
the learning process such as linear decay, exponential decay or any other non-linear function to make the decay (reduction) of the learning rate 
more optimized. Alternatively - you can leave it all to Keras and only update the learning rate if need be, 
through the ReduceLROnPlateau() callback, which, depending on its patience, will reduce the learning rate only when required. 
This is a pretty simple rule and it works surprisingly well! Though, it can be outperformed by other techniques as well. 
Keras offers a LearningRateScheduler class, to which you can pass any function that returns the learning rate, on each epoch, 
and use that function as your custom learning rate scheduler.

    model_static = build_model()
    model_static.summary()
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath='intel-classification_staticlr.h5', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(),
    ]
    
    history = model_static.fit(train_generator,
                        validation_data = valid_generator,
                        callbacks = callbacks,
                        epochs = 1)


