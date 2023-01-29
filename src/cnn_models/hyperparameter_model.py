import tensorflow as tf
import config
from cnn_models.cnn_model import CNNModel
from data_augmentation_util import Rotate90Randomly
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import keras_tuner as kt
from tensorflow import keras


# Code adapted from Keras Hyperparameter Tuning Tutorial:
# Title: Hyperparameter Tuning with KerasTuner and TensorFlow
# Author: Luke Newman
# URL: https://towardsdatascience.com/hyperparameter-tuning-with-kerastuner-and-tensorflow-c4a4d690b31a
# Date Accessed: 09/01/23

class HypModel:

    def __init__(self, train_dataset, validation_dataset):
        self.model_name = config.model
        self.history = None
        self.prediction = None
        self.base_model = None
        self.h_model = None
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.patience_frozen = int(config.max_epoch_frozen/10)
        
        # Instantiate the tuner 
        # TODO: Change for bayesian/random search?
        self.tuner = kt.Hyperband(self.build_model,
                     objective="val_accuracy",
                     max_epochs=config.max_epoch_frozen,
                     factor=3,
                     hyperband_iterations=1,
                     directory="kt_dir",
                     project_name="kt_hyperband4",)
        
        self.tuner.search_space_summary
        
    def run_hyp_pipeline(self):
        f = open(config.output_path + config.name + ".txt", "a")
        f.write(config.name + '\n')
        
        best_hyps = self.hyperparam_search()
        models = self.tuner.get_best_models(num_models=2)
        best_model = models[0]
        # Build the model to show summary of fully connected layers.
        # Needed for `Sequential` without specified `input_shape`.
        best_model.build(input_shape=(None, 28, 28))
        best_model.summary()

        
        self.tuner.results_summary()
        
        history = self.train_best_model(best_hps=best_hyps)
         # training accuracy and loss - plot somewhere else
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        f.write("Training Accuracy:" + str(acc) + '\n')
        print("Training Accuracy:" + str(acc))

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        cnn_model = CNNModel(self.train_dataset)
        cnn_model.accuracy_loss_plots(acc, val_acc, loss, val_loss, "frozen")
        
        history_fine = self.fine_tune_model()
        # Fine-tuned training loss and accuracy
        acc += history_fine.history['accuracy']
        val_acc += history_fine.history['val_accuracy']

        loss += history_fine.history['loss']
        val_loss += history_fine.history['val_loss']

        cnn_model.accuracy_loss_plots(acc, val_acc, loss, val_loss, "unfrozen")

        f.close()
        
        
    def build_model(self, hp):
        """
        Builds model and sets up hyperparameter space to search.
        
        Parameters
        ----------
        hp : HyperParameter object
            Configures hyperparameters to tune.
            
        Returns
        -------
        model : keras model
            Compiled model with hyperparameters to tune.
        """
        
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal', seed=config.RANDOM_SEED),
            tf.keras.layers.RandomFlip('vertical', seed=config.RANDOM_SEED),
            # tf.keras.layers.RandomRotation(0.2, seed=config.RANDOM_SEED,  fill_mode="constant"),
            # tf.keras.layers.RandomZoom(0.2,0.2, seed=config.RANDOM_SEED, fill_mode="constant")
            Rotate90Randomly()
        ])   
        
        # rescale images into range [-1,1] 
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

        # import pre-trained mobile net v2 model using weights learned from training on imagenet
        # include_top=False excludes the classification layers (better for feature extraction)
        IMG_SHAPE = config.IMG_SIZE + (3,)
        
        if self.model_name == "MobileNet":
            self.base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
        elif self.model_name == "VGG":
            self.base_model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
        elif self.model_name == "ResNet":
            self.base_model = tf.keras.applications.ResNet50V2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
        elif self.model_name == "Inception":
            self.base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
        elif self.model_name == "DenseNet":
            self.base_model = tf.keras.applications.DenseNet121(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
        else:
            print("Model not found")
            exit()
        
        image_batch, label_batch = next(iter(self.train_dataset))

        feature_batch = self.base_model(image_batch)
        print(feature_batch.shape)

        # freeze the base model
        self.base_model.trainable = False
        # base_model.summary()

        
        
        # Initialize sequential API and start building model.
        flatten_layer = tf.keras.layers.Flatten()
        fully_connected = tf.keras.Sequential()
        
        # Tune the number of hidden layers and units in each.
        # Number of hidden layers: 1 - 3
        # Number of Units: 32 - 512 with stepsize of 32
        for i in range(0, hp.Int("num_layers", 1, 3)):
            fully_connected.add(
                tf.keras.layers.Dense(
                    units=hp.Int("units_" + str(i), min_value=128, max_value=512, step=32),
                    activation="relu")
                )
            
            # Tune dropout layer with values from 0 - 0.4 with stepsize of 0.1.
            fully_connected.add(tf.keras.layers.Dropout(hp.Float("dropout_" + str(i), 0, 0.4, step=0.1)))
        
        
        # Final output layer that uses sigmoid activation function 
        fully_connected.add(tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer="random_uniform", name='Output'))


        # build the model
        # training false to keep batch normalisation layer in inference mode when we fine-tune
        if config.data_augmentation:
            print("using data augmentation")
            inputs = tf.keras.Input(shape=(config.image_size, config.image_size, 3))
            x = data_augmentation(inputs)
            x = preprocess_input(x)
            x = self.base_model(x, training=False)
            x = flatten_layer(x)
            outputs = fully_connected(x)
            model = tf.keras.Model(inputs, outputs)
        else:
            inputs = tf.keras.Input(shape=(config.image_size, config.image_size, 3))
            x = preprocess_input(inputs)
            x = self.base_model(x, training=False)
            x = flatten_layer(x)
            outputs = fully_connected(x)
            model = tf.keras.Model(inputs, outputs)
        
        hp_optimizer = hp.Choice("optimizer", values=["adam", "sgd", "adadelta", "rmsprop"])
         # Tune learning rate for Adam optimizer with values from 0.001, 0.0001, or 0.00001
        hp_learning_rate = hp.Choice("learning_rate", values=[1e-3, 1e-4, 1e-5])
        
        
        if hp_optimizer == "adam":
            hp_optimizer = tf.keras.optimizers.Adam(hp_learning_rate)
        elif hp_optimizer == "sgd":
            hp_optimizer = tf.keras.optimizers.SGD(hp_learning_rate)
        elif hp_optimizer == "adadelta":
            hp_optimizer = tf.keras.optimizers.Adadelta(hp_learning_rate)
        elif hp_optimizer == "rmsprop":
            hp_optimizer = tf.keras.optimizers.RMSprop(hp_learning_rate)


       
        # Define optimizer, loss, and metrics
        model.compile(optimizer=hp_optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                metrics=['accuracy'])
        
        return model
    
    def hyperparam_search(self):
        
        # Same arguments as model.fit()
        self.tuner.search(
                    self.train_dataset,
                    epochs=config.max_epoch_frozen,
                    validation_data=self.validation_dataset,
                    callbacks=[
                        EarlyStopping(monitor='val_loss', patience=self.patience_frozen, restore_best_weights=True),
                        ReduceLROnPlateau(patience=int(self.patience_frozen / 2)) 
                    ])  
        
        # Get the optimal hyperparameters from the results
        best_hps=self.tuner.get_best_hyperparameters()[0]
        
        return best_hps
    
    def train_best_model(self, best_hps):
        self.h_model = self.tuner.hypermodel.build(best_hps)
        
        loss0, accuracy0 = self.h_model.evaluate(self.validation_dataset)

        print("initial loss: {:.2f}".format(loss0))
        print("initial accuracy: {:.2f}".format(accuracy0))
        
        # Train hypertuned model
        history = self.h_model.fit(
                self.train_dataset,
                epochs=config.max_epoch_frozen,
                validation_data=self.validation_dataset,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=self.patience_frozen, restore_best_weights=True),
                    ReduceLROnPlateau(patience=int(self.patience_frozen / 2)) 
                ])  
        
        return history
    
    def fine_tune_model(self):
        # unfreeze top layers for fine-tuning 
        self.base_model.trainable = True

        # fine-tune with a small preset learning rate
        self.h_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])
        
        max_fine_tune_epochs = config.max_epoch_unfrozen
        fine_tune_patience = int(max_fine_tune_epochs/10) #10 before, 5 after

        history_fine = self.h_model.fit(self.train_dataset,
                                epochs=max_fine_tune_epochs,
                                validation_data=self.validation_dataset,
                                callbacks=[
                                    EarlyStopping(monitor='val_loss', patience=fine_tune_patience, restore_best_weights=True),
                                    ReduceLROnPlateau(patience=int(fine_tune_patience / 2)) 
                                ])

        return history_fine
    
        # Modified from Craig's code
    # https://github.com/CraigMyles/cggm-mammography-classification/blob/main/4_Results_Accuracy_By_Model.ipynb
    def evaluate_model(self, test_dataset, y_true):
        #Write output to a file with same name as plot
        f = open(config.output_path + config.name + ".txt", "a")
        f.write(config.name + '\n')
        
        loss, accuracy = self.h_model.evaluate(test_dataset)

        predictions_list = np.array([])
        labels = np.array([])
        for x, y in test_dataset:
            pred = self.h_model.predict(x)
            predictions_list = np.concatenate([predictions_list,((pred > 0.5)+0).ravel()])
            labels = np.concatenate([labels, y.numpy().flatten()])
        
        report = classification_report(labels, predictions_list, target_names = ['Benign (Class 0)','Malignant (Class 1)'])
        print(report)
        f.write(report + '\n')
        
        
        #Confusion Matrix
        class_labels = ['Benign', 'Malignant']
        cm = confusion_matrix(labels, predictions_list)
        
        disp = ConfusionMatrixDisplay(cm, display_labels=class_labels)
        plt.figure()
        disp.plot()
        plt.savefig(config.output_path + config.name + "-confusion-matrix.png")

        auc_score = roc_auc_score(y_true, predictions_list)
        print("AUC:"+ str(auc_score))
        f.write("AUC:"+str(auc_score)+ '\n')
        
        print('Test accuracy :', accuracy)
        f.write('Test accuracy :' + str(accuracy) + '\n')
        print('Test loss :', loss) 
        f.write('Test loss :' + str(loss)+ '\n') 
        f.close()
        
    def save_model(self):
        self.h_model.save(config.saved_models + config.name + '.h5')
