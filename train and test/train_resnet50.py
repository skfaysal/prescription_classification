from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.python.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import glob
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.preprocessing.image import img_to_array
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from imutils import paths
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
from imutils import paths
import seaborn as sns


# validation can be added from keras pipeline

class TrainingLeftRight:
    def __init__(self, img_rows, img_cols, epochs, batch_size, validation_split):
        self.train_data_path = str(os.getcwd()) + '/' + 'TrainData/' + 'train'
        self.val_data_path = str(os.getcwd()) + '/' + 'TrainData/' + 'validation'
        self.test_data_path = str(os.getcwd()) + '/' + 'TrainData/' + 'test'
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.epochs = epochs
        self.batch_size = batch_size
        self.INIT_LR = 1e-4
        self.validation_split = validation_split

    @staticmethod
    def prepare_csv(self):
        # For train data
        train_labels = []
        for img in os.listdir(self.train_data_path):
            if 'left' in img:
                train_labels.append((img, 0))
            elif 'right' in img:
                train_labels.append((img, 1))
        # print(labels)
        labels = pd.DataFrame(train_labels, columns=['name', 'label'])
        labels = pd.DataFrame(train_labels, columns=['name', 'label'], dtype=str)
        num_of_train_samples = len(train_labels)
        labels.to_csv('train_label.csv', index=False)
        train_df = pd.read_csv('train_label.csv', dtype=str)

        # For validation data
        val_labels = []
        for img in os.listdir(self.val_data_path):
            if 'left' in img:
                val_labels.append((img, 0))
            elif 'right' in img:
                val_labels.append((img, 1))
        labels = pd.DataFrame(val_labels, columns=['name', 'label'])
        labels = pd.DataFrame(val_labels, columns=['name', 'label'], dtype=str)
        num_of_val_samples = len(val_labels)
        labels.to_csv('val_label.csv', index=False)
        val_df = pd.read_csv('val_label.csv', dtype=str)

        # For test data
        test_labels = []
        for img in os.listdir(self.test_data_path):
            if 'left' in img:
                test_labels.append((img, 0))
            elif 'right' in img:
                test_labels.append((img, 1))
        labels = pd.DataFrame(test_labels, columns=['name', 'label'])
        labels = pd.DataFrame(test_labels, columns=['name', 'label'], dtype=str)
        num_of_test_samples = len(test_labels)
        labels.to_csv('test_label.csv', index=False)
        test_df = pd.read_csv('test_label.csv', dtype=str)
        return train_df, val_df, test_df

    @staticmethod
    def data_generator_from_dataframe(self):
        train_df, valid_df, test_df = TrainingLeftRight.prepare_csv(self)
        Datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                     validation_split=self.validation_split,
                                     rotation_range=360,
                                     horizontal_flip=True,
                                     vertical_flip=True)

        train_generator = Datagen.flow_from_dataframe(dataframe=train_df,

                                                      directory=self.train_data_path,
                                                      x_col="name",
                                                      y_col="label",
                                                      target_size=(self.img_rows, self.img_cols),
                                                      batch_size=self.batch_size,
                                                      class_mode="categorical")

        validation_generator = Datagen.flow_from_dataframe(dataframe=valid_df,
                                                           directory=self.val_data_path,
                                                           x_col="name",
                                                           y_col="label",
                                                           target_size=(self.img_rows, self.img_cols),
                                                           batch_size=self.batch_size,
                                                           class_mode="categorical")

        test_generator = Datagen.flow_from_dataframe(dataframe=test_df,
                                                     directory=self.test_data_path,
                                                     x_col="name",
                                                     y_col="label",
                                                     target_size=(self.img_rows, self.img_cols),
                                                     batch_size=1,
                                                     class_mode="categorical",
                                                     shuffle=False)
        return train_generator, validation_generator, test_generator

    def data_generator_from_directory(self):
        Datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                     validation_split=self.validation_split,
                                     rotation_range=360,
                                     horizontal_flip=True,
                                     vertical_flip=True)


        train_generator = Datagen.flow_from_directory(directory=self.train_data_path,
                                                      target_size=(self.img_rows, self.img_cols),
                                                      batch_size=self.batch_size,
                                                      class_mode="categorical",
                                                      subset='training')

        validation_generator = Datagen.flow_from_directory(directory=self.train_data_path,
                                                           target_size=(self.img_rows, self.img_cols),
                                                           batch_size=self.batch_size,
                                                           class_mode="categorical",
                                                           subset='validation')

        test_generator = Datagen.flow_from_directory(directory=self.test_data_path,
                                                     target_size=(self.img_rows, self.img_cols),
                                                     batch_size=1,
                                                     class_mode="categorical",
                                                     shuffle=False)
        print(test_generator.class_indices)


        return train_generator, validation_generator, test_generator

    def t_gen(self, model):
        test_data = ImageDataGenerator(preprocessing_function=preprocess_input)
        t_generator = test_data.flow_from_directory(directory=self.test_data_path,
                                                       target_size=(self.img_rows, self.img_cols),
                                                       batch_size=1,
                                                       class_mode="categorical",
                                                       shuffle=False)
        print(".........Using t_gen.........")
        predictions = model.predict(t_generator, batch_size=1)
        print(classification_report(t_generator.classes,
                                    predictions.argmax(axis=1)))

        cm = confusion_matrix(t_generator.classes,
                               predictions.argmax(axis=1))
        print(cm)
        labels = list(t_generator.class_indices.keys())
        print(labels)
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax, cmap='Blues')

        # labels, title and ticks
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)
        ax.figure.savefig(str(os.getcwd())+'/output/'+'confusion matrix.png')
        plt.close()

    def single_image(self, model, test_generator):
        img_data__path = list(paths.list_images(self.test_data_path))
        pred_list = []
        label_list = []
        for path in img_data__path:
            label = path.split(os.path.sep)[-2]
            
            if 'Not Pres' in label:
                label_list.append(0)
            elif 'Pres' in label:
                label_list.append(1)
            original = load_img(path, target_size=(224, 224))
            numpy_image = img_to_array(original)
            image_batch = np.expand_dims(numpy_image, axis=0)
            processed_image = preprocess_input(image_batch.copy())
            prediction = model.predict(processed_image)
            prediction = prediction.argmax(axis=1)
            pred_list.append(prediction)

        pred_array = np.array(pred_list)
        predict = pred_array.reshape(-1, 1)
        img_label = np.array(label_list)

        print(".........Using single_image.........")
        print(classification_report(img_label,predict))

        cm = confusion_matrix(img_label,predict,)
        print(cm)




    def resnet50(self):
        train_generator, validation_generator, test_generator = TrainingLeftRight.data_generator_from_directory(self)

        # load the ResNet-50 network, ensuring the head FC layer sets are left off
        baseModel = ResNet50(weights="imagenet", include_top=False,
                             input_tensor=Input(shape=(self.img_rows, self.img_cols, 3)))

        # construct the head of the model that will be placed on top of the
        # the base model
        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(64, activation="relu")(headModel)
        headModel = Dropout(0.2)(headModel)
        headModel = Dense(32, activation="relu")(headModel)
        headModel = Dropout(0.2)(headModel)
        headModel = Dense(2, activation="softmax")(headModel)

        # place the head FC model on top of the base model (this will become
        # the actual model we will train)
        model = Model(inputs=baseModel.input, outputs=headModel)

        # loop over all layers in the base model and freeze them so they will
        # *not* be updated during the first training process
        for layer in baseModel.layers:
            layer.trainable = False

        # compile our model
        print("[INFO] compiling model...")
        opt = Adam(lr=self.INIT_LR, decay=self.INIT_LR / self.epochs)
        model.compile(loss="categorical_crossentropy", optimizer=opt,
                      metrics=["accuracy"])
        # Set callback functions to early stop training and save the best model so far
        # callbacks = [EarlyStopping(monitor='val_loss', patience=5),
        #              ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

        history = model.fit_generator(train_generator,
                                      steps_per_epoch=train_generator.samples // self.batch_size,
                                      epochs=self.epochs,
                                      validation_data=validation_generator,
                                      validation_steps=validation_generator.samples // self.batch_size)
        # evaluate the network
        print(".........[FYI]evaluating network using test gen.........")
        predictions = model.predict(test_generator, batch_size=1)
        print(classification_report(test_generator.classes,
                                    predictions.argmax(axis=1)))

        print(confusion_matrix(test_generator.classes,
                               predictions.argmax(axis=1)))


        model.save_weights(str(os.getcwd())+'/output/'+'Weights_binary_resnet50.h5')
        model.save(str(os.getcwd())+'/output/'+'model_binary_resnet50.h5')

        # list all data in history
        print(history.history)
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(str(os.getcwd())+'/output/'+'model_accuracy.png')
        plt.close()

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(str(os.getcwd())+'/output/'+'model_loss.png')
        plt.close()
        # call
        TrainingLeftRight.t_gen(self, model)

        # call 2
        TrainingLeftRight.single_image(self, model, test_generator)


ap = argparse.ArgumentParser()
ap.add_argument("-r", "--img_row", required=True, help="path to input dr model")
ap.add_argument("-c", "--img_col", required=True, help="path to input left_right model")
ap.add_argument("-e", "--epochs", required=True, help="path to input image_data")
ap.add_argument("-b", "--batch_size", required=True, help="path to input csv_data")
ap.add_argument("-val", "--val_split", required=True, help="path to input folder")
args = vars(ap.parse_args())

if __name__ == '__main__':
    obj = TrainingLeftRight(int(args["img_row"]), int(args["img_col"]), int(args["epochs"]), int(args["batch_size"]),
                            float(args["val_split"]))
    # train_generator, validation_generator, test_generator = obj.data_generator_from_directory()
    obj.resnet50()
