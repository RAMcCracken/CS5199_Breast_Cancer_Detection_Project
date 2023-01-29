import config
import tensorflow as tf
import matplotlib.pyplot as plt
from data_augmentation_util import Rotate90Randomly

def create_model(model_name, train_dataset):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal', seed=config.RANDOM_SEED),
        tf.keras.layers.RandomFlip('vertical', seed=config.RANDOM_SEED),
        Rotate90Randomly()
    ])     
    
    for image, _ in train_dataset.take(1):
            first_image = image[0]
            plt.figure()
            plt.imshow(first_image/255)
            plt.savefig(config.output_path + 'chosen_image2.png')
            
            plt.figure(figsize=(10,10))
            for i in range(9):
                ax = plt.subplot(3,3,i+1)
                augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
                plt.imshow(augmented_image[0]/255)
                plt.axis('off')
                plt.savefig(config.output_path + 'sample_augmentation_trial2.png')
    
    # rescale images into range [-1,1] 
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    # import pre-trained mobile net v2 model using weights learned from training on imagenet
    # include_top=False excludes the classification layers (better for feature extraction)
    IMG_SHAPE = config.IMG_SIZE + (3,)
    
    if model_name == "MobileNet":
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    elif model_name == "VGG":
        base_model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    elif model_name == "ResNet":
        base_model = tf.keras.applications.ResNet50V2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    elif model_name == "Inception":
        base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    elif model_name == "DenseNet":
        base_model = tf.keras.applications.DenseNet121(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    else:
        print("Model not found")
        exit()
    
    image_batch, label_batch = next(iter(train_dataset))

    feature_batch = base_model(image_batch)
    print(feature_batch.shape)

    # freeze the base model
    base_model.trainable = False
    # base_model.summary()

    flatten_layer = tf.keras.layers.Flatten()
    
    fully_connected = tf.keras.Sequential()
    # Fully connected layers.
    fully_connected.add(tf.keras.layers.Dropout(0.4, seed=config.RANDOM_SEED, name="Dropout_1"))
    fully_connected.add(tf.keras.layers.Dense(units=512, activation='relu', name='Dense_1'))
    # fully_connected.add(tf.keras.layers.Dropout(0.2, seed=config.RANDOM_SEED, name="Dropout_2")) #Added for trial 
    fully_connected.add(tf.keras.layers.Dense(units=32, activation='relu', name='Dense_2'))

    # Final output layer that uses sigmoid activation function 
    fully_connected.add(tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer="random_uniform", name='Output'))

    # build the model
    # training false to keep batch normalisation layer in inference mode when we fine-tune
    if config.data_augmentation:
        print("using data augmentation")
        inputs = tf.keras.Input(shape=(config.image_size, config.image_size, 3))
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = base_model(x, training=False)
        x = flatten_layer(x)
        outputs = fully_connected(x)
        model = tf.keras.Model(inputs, outputs)
    else:
        inputs = tf.keras.Input(shape=(config.image_size, config.image_size, 3))
        x = preprocess_input(inputs)
        x = base_model(x, training=False)
        x = flatten_layer(x)
        outputs = fully_connected(x)
        model = tf.keras.Model(inputs, outputs)
    
    return model, base_model