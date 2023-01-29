import cnn_models.model_builder as model_builder
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import config
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import config
import time

class CNNModel:
    
    def __init__(self, train_dataset, *, testmode=False):
        self.model_name = config.model
        self.history = None
        self.prediction = None
        
        if not testmode:
            self.model, self.base_model = model_builder.create_model(self.model_name, train_dataset)
        else:
            # In test mode load the model given in the -n name parameter 
            self.model =tf.keras.models.load_model(config.saved_models + config.name + ".h5")
    
    def compile_model(self, learning_rate):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])
        
        
    def train_model_frozen(self, train_dataset, validation_dataset):
        self.compile_model(config.learning_rate)
        
        max_epochs = config.max_epoch_frozen
        patience = int(max_epochs/10) 

        loss0, accuracy0 = self.model.evaluate(validation_dataset)

        print("initial loss: {:.2f}".format(loss0))
        print("initial accuracy: {:.2f}".format(accuracy0))
        
        history = self.model.fit(train_dataset,
                    epochs=max_epochs,
                    validation_data=validation_dataset,
                    callbacks=[
                        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
                        ReduceLROnPlateau(patience=int(patience / 2)) 
                    ])

        return history
        
    def fine_tune_model(self, train_dataset, validation_dataset):
        # unfreeze layers for fine-tuning 
        self.base_model.trainable = True

        self.compile_model(config.fine_tune_learning_rate)
        
        max_fine_tune_epochs = config.max_epoch_unfrozen
        fine_tune_patience = int(max_fine_tune_epochs/10) 

        history_fine = self.model.fit(train_dataset,
                                epochs=max_fine_tune_epochs,
                                validation_data=validation_dataset,
                                callbacks=[
                                    EarlyStopping(monitor='val_loss', patience=fine_tune_patience, restore_best_weights=True),
                                    ReduceLROnPlateau(patience=int(fine_tune_patience / 2)) 
                                ])

        return history_fine
    
    def accuracy_loss_plots(self, acc, val_acc, loss, val_loss, label):
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0,1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.show()
        plt.savefig(config.output_path + config.name + "-" + label + '-train.png')
        
    
    # Modified from Craig's code
    # https://github.com/CraigMyles/cggm-mammography-classification/blob/main/4_Results_Accuracy_By_Model.ipynb
    def evaluate_model(self, test_dataset):
        start_time = time.time()
        
        #Write output to a file with same name as plot
        f = open(config.output_path + config.name + "-test.txt", "a")
        f.write(config.name + '\n')
        
        print("-- Testing Model --")
        f.write("-- Testing Model --")
        f.write(config.name)
        print(config.name)
        
        loss, accuracy = self.model.evaluate(test_dataset)

        time_elapsed = time.time() - start_time

        predictions_list = np.array([])
        labels = np.array([])
        for x, y in test_dataset:
            pred = self.model.predict(x)
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
        plt.savefig(config.output_path + config.name + "-confusion-matrix-test.png")

        auc_score = roc_auc_score(labels, predictions_list)
        print("AUC:"+ str(auc_score))
        f.write("AUC:"+str(auc_score)+ '\n')
        
        print('Test accuracy :', accuracy)
        f.write('Test accuracy :' + str(accuracy) + '\n')
        print('Test loss :', loss) 
        f.write('Test loss :' + str(loss)+ '\n') 
        
        print("Testing Runtime: " + str(time_elapsed) + " seconds")
        f.write("Testing Runtime: " + str(time_elapsed) + " seconds \n")
        f.close
        
        
    def save_model(self):
        self.model.save(config.saved_models + config.name + '.h5')

        
