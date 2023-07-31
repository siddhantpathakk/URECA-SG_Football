import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import constants
import numpy as np


class LogReg_DL:
    """
    Class for Logistic Regression modelling using DL (ANNs)
    """

    def __init__(self, _Xtrain, _ytrain, _Xtest, _ytest):
        self.X_train, self.y_train, self.X_test, self.y_test = (
            _Xtrain,
            _ytrain,
            _Xtest,
            _ytest,
        )
        self.y_pred, self.history, self.model, self.optimizer = None, None, None, None
        print("The model has been initialised.")

    def create_model(self):
        """
        Creates a sequential layer model (using tensorflow) and adds layers
        Prints the summary of the model at the end
        """
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(len(self.X_train.columns),)))
        
        # self.model.add(Dense(units=200, activation=tf.nn.elu))
        self.model.add(Dense(units=200, activation="relu"))
        self.model.add(Dense(units=200, activation="relu"))
        self.model.add(Dense(units=200, activation="relu"))
        self.model.add(Dense(units=200, activation="relu"))
        # self.model.add(Dense(units=200, activation=tf.nn.elu))
        # self.model.add(Dense(units=200, activation=tf.nn.elu))


        self.model.add(Dense(units=1, activation="sigmoid"))
        print("The model has been created.")
        self.model.summary()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=constants.LEARNING_RATE)

    def fit(self):
        """
        Compiles the custom-defined ANN along with the optimizer and fits it
        """
        self.model.compile(
            optimizer=self.optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=constants.NO_EPOCHS,
            batch_size=constants.BATCH_SIZE,
            validation_split=constants.VAL_SPLIT,
            shuffle=True,
        )
        print("The model has been fit.")
        self.model.save("Inferences/Result Logs/Model Versions/my_ANN_logistic_regression.h5")

    def accuracy(self, PATH_TO_STORE):
        """
        Plots the model accuracy-epoch graph and model loss-epoch graph
        Input: a path to the directory where the graphs should be stored
        """

        def accuracy_plot(self):
            plt.plot(self.history.history["accuracy"], label="Train")
            plt.plot(self.history.history["val_accuracy"], label="Validation")
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.title("Model Accuracy")
            plt.legend(loc="upper left")
            # plt.show()
            plt.savefig(PATH_TO_STORE + "/model_accuracy.png")
            print(
                "Model Accuracy graph saved at " + PATH_TO_STORE + "/model_accuracy.png"
            )
            plt.close()

        def loss_plot(self):
            plt.plot(self.history.history["loss"], label="Train")
            plt.plot(self.history.history["val_loss"], label="Validation")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.title("Model Loss")
            plt.legend(loc="upper right")
            # plt.show()
            plt.savefig(PATH_TO_STORE + "/model_loss.png")
            print("Model Loss graph saved at " + PATH_TO_STORE + "/model_loss.png")

        accuracy_plot(self)
        loss_plot(self)
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)

        print("Test loss:", test_loss)
        print("Test accuracy:", test_acc)
