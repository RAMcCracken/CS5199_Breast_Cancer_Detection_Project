from pip import main
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from data_preparation.import_data import import_cbisddsm_training_dataset
from cnn_models.cnn_model import CNNModel
from cnn_models.hyperparameter_model import HypModel
import numpy as np
import sys
import time
import config
import data_preprocessing.shared_preprocessing as preprocessing

BATCH_SIZE = config.batch_size
tf.random.set_seed(config.RANDOM_SEED)

def load_DDSM_data():
    ddsm_dir = "../../data/ddsm/"
    if config.preproc:
        parent = "PNG-PREPROC/"
    else:
        parent = "PNG/"

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        ddsm_dir + parent + "TRAIN/ALL",
        labels="inferred",
        label_mode="binary",
        image_size=config.IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True, 
        seed=config.RANDOM_SEED)

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        ddsm_dir + parent + "TEST/ALL",
        labels="inferred",
        label_mode="binary",
        image_size=config.IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=config.RANDOM_SEED
    )
    
    test_y_true = np.concatenate([y for x, y in test_dataset], axis=0)

    # Create test set from splitting validation set
    train_batches = tf.data.experimental.cardinality(train_dataset)
    validation_dataset = train_dataset.take(train_batches // 5)
    train_dataset = train_dataset.skip(train_batches // 5)

    val_y_true = np.concatenate([y for x, y in validation_dataset], axis=0)
    
    return train_dataset, validation_dataset, test_dataset, test_y_true, val_y_true
    
def load_CMMD_data():
    cmmd_dir = "../../data/cmmd/"
    if config.preproc:
        parent = "PNG-PREPROC/"
    else:
        parent = "PNG/"
    
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        cmmd_dir+ parent +"TRAIN/ALL",
        labels="inferred",
        label_mode="binary",
        image_size=config.IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=config.RANDOM_SEED)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        cmmd_dir + parent + "VAL/ALL",
        labels="inferred",
        label_mode="binary",
        image_size=config.IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=config.RANDOM_SEED
    )
    
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        cmmd_dir + parent + "TEST/ALL",
        labels="inferred",
        label_mode="binary",
        image_size=config.IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=config.RANDOM_SEED)

    test_y_true = np.concatenate([y for x, y in test_dataset], axis=0)
    val_y_true = np.concatenate([y for x, y in validation_dataset], axis=0)
    
    
    return train_dataset, validation_dataset, test_dataset, test_y_true, val_y_true
    
def combine_DDSM_and_CMMD():
    train_dataset_ddsm, validation_dataset_ddsm, test_dataset_ddsm, test_y_true_ddsm, val_y_true_ddsm = load_DDSM_data()
    train_dataset_cmmd, validation_dataset_cmmd, test_dataset_cmmd, test_y_true_cmmd, val_y_true_cmmd = load_CMMD_data()
    
    # join datasets together 
    train_dataset_both = tf.data.Dataset.concatenate(train_dataset_ddsm, train_dataset_cmmd)
    train_dataset_both = train_dataset_both.shuffle(BATCH_SIZE, seed = config.RANDOM_SEED, reshuffle_each_iteration=False)

    validation_dataset_both = tf.data.Dataset.concatenate(validation_dataset_ddsm, validation_dataset_cmmd)
    validation_dataset_both = validation_dataset_both.shuffle(BATCH_SIZE, seed = config.RANDOM_SEED, reshuffle_each_iteration=False)
    
    test_dataset_both = tf.data.Dataset.concatenate(test_dataset_ddsm, test_dataset_cmmd)
    test_dataset_both = test_dataset_both.shuffle(BATCH_SIZE, seed = config.RANDOM_SEED, reshuffle_each_iteration=False)
    
    # need val and test y sets as well
    test_y_true = np.concatenate([y for x, y in test_dataset_both], axis=0)
    val_y_true = np.concatenate([y for x, y in validation_dataset_both], axis=0)
    
    return train_dataset_both, validation_dataset_both, test_dataset_both, test_y_true, val_y_true


def main():
    
    # Preprocess files if preproc flag set and load datasets
    if config.dataset == 'DDSM':
        # Include if preprocessing pipeline has not been run and images need to be created for the first time
        # if config.preproc:
        #     preprocessing.run_pipeline(config.dataset)
        train_dataset, validation_dataset, test_dataset, test_y_true, val_y_true = load_DDSM_data()
    elif config.dataset == 'CMMD':
        # Include if preprocessing pipeline has not been run and images need to be created for the first time
        # if config.preprocessing:
        #     preprocessing.run_pipeline(config.dataset)
        train_dataset, validation_dataset, test_dataset, test_y_true, val_y_true = load_CMMD_data()
    elif config.dataset == 'BOTH':
        # Include if preprocessing pipeline has not been run and images need to be created for the first time
        # if config.preproc:
        #     preprocessing.run_pipeline('DDSM')
        #     preprocessing.run_pipeline('CMMD')
        train_dataset, validation_dataset, test_dataset, test_y_true, val_y_true = combine_DDSM_and_CMMD()
    
    # Dataset IO buffering for performance
    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    
    plt.figure(figsize=(10, 10))
    for images, labels in train_dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.axis("off")
            plt.savefig(config.output_path + 'sample.png')    
            
    if config.test_mode:
        # In test mode - program loads a saved trained model and evaluates it on the unseen test set 
        cnn_model = CNNModel(train_dataset, testmode=config.test_mode)
    
        cnn_model.evaluate_model(test_dataset)
        
    elif config.hyperparam_tuning:
        # Perform a hyperparameter tuning search and evaluate the chosen model
        hyp_model = HypModel(train_dataset, validation_dataset)
        hyp_model.run_hyp_pipeline()
        hyp_model.evaluate_model(validation_dataset, val_y_true)
        hyp_model.save_model()
    else:
        cnn_model = CNNModel(train_dataset)

        # compile model
        history = cnn_model.train_model_frozen(train_dataset, validation_dataset)

        # training accuracy and loss - plot somewhere else
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        # Create plots of training and validation loss and accuracy after fine-tuning
        cnn_model.accuracy_loss_plots(acc, val_acc, loss, val_loss, "frozen")

        history_fine = cnn_model.fine_tune_model(train_dataset, validation_dataset)

        # Fine-tuned training loss and accuracy
        acc += history_fine.history['accuracy']
        val_acc += history_fine.history['val_accuracy']

        loss += history_fine.history['loss']
        val_loss += history_fine.history['val_loss']

        # Create plots of training and validation loss and accuracy after fine-tuning
        cnn_model.accuracy_loss_plots(acc, val_acc, loss, val_loss, "unfrozen")

        cnn_model.evaluate_model(validation_dataset)
        cnn_model.save_model()

    
    
if __name__ == "__main__":
    #Pattern for parsing arguments used from Adam Jaamour with modification by Rhona McCracken
    #https://github.com/Adamouization/Breast-Cancer-Detection-Mammogram-Deep-Learning-Publication/blob/main/src/main.py
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-d", "--dataset",
                        default="DDSM",
                        required=True,
                        choices=["DDSM", "CMMD", "BOTH"],
                        help="Dataset to use. Must be one of: 'DDSM', 'CMMD' or 'BOTH'") 
    parser.add_argument("-t", "--test",
                        action="store_true",
                        default=False, 
                        help="Include this flag to run program in testmode - test the model entered with name parameter")
    parser.add_argument("-p", "--preprocessing",
                        action="store_true",
                        default=False,
                        help="Include this flag to enable the pre-processing pipeline (not including data augmentation)")
    parser.add_argument("-a", "--augmentation",
                        action="store_true",
                        default=False,
                        help="Include this flag to enable data augmentation")
    parser.add_argument("-hy", "--hyperparam_tuning",
                        action="store_true",
                        default=False,
                        help="Include this flag to enable hyperparameter tuning")
    parser.add_argument("-lr", "--learning_rate", 
                        type=float, 
                        default=1e-4, 
                        help="Learning Rate for additional unfrozen layers added to the pre-trained network for transfer learning. Defaults to 1e-4")
    parser.add_argument("-m", "--model",
                        default="MobileNet", 
                        choices=["MobileNet", "VGG", "ResNet", "Inception", "DenseNet"]) #TODO: check this list in next investigation 
    parser.add_argument("-s", "--image_size",
                        type=int,
                        default=160,
                        help="Image size for resizing input images as squares, default 160 x 160 pixels"
                        )
    parser.add_argument("-n", "--name",
                        default="Sample",
                        help="The name of the experiment being trained/tested. Defaults to string: Sample.")
    parser.add_argument("-b", "--batch_size",
                        default=32,
                        type=int,
                        help="Batch size, defaults to 32")
    parser.add_argument("-e","--fine_tune_epochs",
                        default=50,
                        type=int,
                        help="Max number of epochs for fine-tuning")
    
    
    args = parser.parse_args()
    config.dataset = args.dataset 
    config.preproc = args.preprocessing
    config.data_augmentation = args.augmentation
    if args.learning_rate <= 0:
        print("Learning rate out of range please specify a learning rate above 0.")
        exit()
    config.learning_rate = args.learning_rate
    config.model = args.model
    config.name = args.name
    config.image_size = args.image_size
    config.IMG_SIZE = (config.image_size, config.image_size)
    config.batch_size = args.batch_size
    config.max_epoch_unfrozen = args.fine_tune_epochs
    config.hyperparam_tuning = args.hyperparam_tuning
    config.test_mode = args.test
    
    main()