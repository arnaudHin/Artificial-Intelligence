

"""
Autor : Arnaud HINCELIN
Classification of images cats/dogs : CNN
Dataset Microsoft cats/dogs : https://www.microsoft.com/en-us/download/details.aspx?id=54765
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2  
import os
import random
import pickle

from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split, validation_curve

#load training datas with pickle library
def preprocessing_data_catsDogs(directory, categories):
	IMG_SIZE = 100
	dataset = []
	for category in categories:  # do dogs and cats
	    path = os.path.join(directory,category)  # create path to dogs and cats
	    class_num = categories.index(category)
	    for img in os.listdir(path):  # iterate over each image per dogs and cats
	    	try:
	    		img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
	    		new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
	    		dataset.append([new_array, class_num])
	    	except Exception as e:
	    		pass
	random.shuffle(dataset)
	X = []
	y = []
	for features, label in dataset:
		X.append(features)
		y.append(label)
	X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

	#array et normalisation
	X = np.array(X)
	y = np.array(y)

	return X, y

def pickle_saving(X, y):
	pickle_out = open("X.pickle","wb")
	pickle.dump(X, pickle_out)
	pickle_out.close()
	pickle_out = open("y.pickle","wb")
	pickle.dump(y, pickle_out)
	pickle_out.close()


def pickle_opening():
	pickle_in = open("X.pickle","rb")
	X = pickle.load(pickle_in)
	pickle_in = open("y.pickle","rb")
	y = pickle.load(pickle_in)

	X = X/255.0

	return X, y

def create_model(X):
	#-------MODEL------
	model = Sequential()
	###CNN 
	#256 filtres avec une fenêtre de 3x3
	#Mais si pas assez précis car images de dim plus gdes / RGB => diminuer fenetre et aug filtres
	model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]) )
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())  # convertir nos 3D feature maps en 1D feature vectors
	model.add(Dense(64))
	#ATTENTION !! Dernière couche doit contenir le même nb de neurones que le nb de Labels !
	model.add(Dense(1)) 
	model.add(Activation('sigmoid'))
	return model

def train_model(model, X, y):
	#------TRAINNIG------
	#optimizer ?  
	#commence par compiler le "reseau"
	#params sur site keras ; metrics (metrique de calcul, manhattan,...), loss (calcul du coût), optimizer (copmliqué...)
	model.compile(
		optimizer='adam', 
		loss='binary_crossentropy', 
		metrics=['accuracy']
		) #loss de crossentropy binaire => classification binaire

	#entrainement du reseau ; ici long car nos couches conv 
	#possèdes bcp de filtres et nos fenetres sont petites + images 100x100
	model.fit(X, y, 
		batch_size=32, 
		epochs=3, 
		validation_split = 0.3
		)
	return model

def evaluate_model(model, x_test, y_test):
	val_loss, val_accuracy = model.evaluate(x_test, y_test)
	print("Cout de : ", val_loss) #environ 0.013
	print("Precision de : ", val_accuracy) #environ 0.502

def predict_result(model, index_test_set, features, labels):
	label_true = labels[index_test_set]
	feature = features[index_test_set].reshape(1, 100, 100, 1)
	predict = model.predict(feature)
	label_pred = np.round(predict)
	print("Le vrai chiffre est (OneHotEncoder) : ", label_true)
	print("Modèle prédit : ", (predict), "donc : " ,label_pred)
	plt.imshow(features[index_test_set])
	plt.show()

def predict_result_extern_data(model, image, label):
	label_true = label
	predict = model.predict(image)
	#label_pred = np.argmax(model.predict(y_test_a[index_test_set]))
	label_pred = np.argmax(predict)
	print("Le vrai chiffre est ", label_true)
	print("Modèle prédit : ", (predict), "donc : " ,label_pred)
	#plt.imshow(image)
	#plt.show()

def  show_results(X):
	plt.figure()
	plt.subplot(2, 3, 1)
	plt.title("Chien (90%)", color='red', size=18)
	plt.imshow(X[30], cmap='gray')
	plt.subplot(2, 3, 2)
	plt.title("Chien (99%)", color='red', size=18)
	plt.imshow(X[77], cmap='gray')
	plt.subplot(2, 3, 3)
	plt.title("Chien (89%)", color='red', size=18)
	plt.imshow(X[1043], cmap='gray')
	plt.subplot(2, 3, 4)
	plt.title("Chat (97%)", color='blue', size=18)
	plt.imshow(X[47], cmap='gray')
	plt.subplot(2, 3, 5)
	plt.title("Chat (80%)", color='blue', size=18)
	plt.imshow(X[89], cmap='gray')
	plt.subplot(2, 3, 6)
	plt.title("Chat (82%)", color='blue', size=18)
	plt.imshow(X[56], cmap='gray')
	plt.show()



###______________MAIN___________### -----------------------------------------------------------------------------

#charger dataset => images 28x28 codées digits de 0à9 
PATH_DIR = "PetImages"
CATEGORIES = ["Dog", "Cat"]

#________________TREAT+CREATE+TRAIN (à faire 1 seule fois)_______________
# Preprocessing des données

"""
X, y = preprocessing_data_catsDogs(PATH_DIR, CATEGORIES)
pickle_saving(X, y)
X, y = pickle_opening()
model = create_model(X)
model = train_model(model, X, y)
model.save("CatDogs_microsoft_TRUE.h5")
"""


#______________INFERENCE___________


X, y = pickle_opening()
model = models.load_model("CatDogs_microsoft_TRUE.h5")

show_results(X)

