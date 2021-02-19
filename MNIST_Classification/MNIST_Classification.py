

"""
Autor : Arnaud HINCELIN
Classification of images handwritten numbers : CNN
Dataset Microsoft cats/dogs : http://yann.lecun.com/exdb/mnist/
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2  

from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split, validation_curve


def preprocessing_data_mnist(x_dev, y_dev, x_test, y_test):
	
	print("x_dev : ", x_dev.shape,"\n") #(60000, 28, 28) 
	print("y_dev : ", y_dev.shape,"\n") #(60000,)
	print("x_test : ", x_test.shape, "\n") #(10000, 28, 28) 
	print("y_test : ", y_test.shape,"\n") #(10000,) 

	#Preprocessing simple : normalisation, on ferait bien plus si les images avaient des dimensions différentes !!
	#Normalisation : mettre à échelle de -1 et 1 selon les extremum de l'array concernée
	x_dev_n = utils.normalize(x_dev, axis=1)
	x_test_n = utils.normalize(x_test, axis=1)

	#Creation du set de validation : 20% du dev set
	x_train_n, x_val_n, y_train, y_val = train_test_split(x_dev_n, y_dev, test_size=0.2, random_state=1)

	#Reshape des features pr avoir bonne dimensions, -1 car infini mais on pourrait mettre len(x_train)...
	x_train_n_r = x_train_n.reshape(-1, 28, 28, 1)
	x_val_n_r = x_val_n.reshape(-1, 28, 28, 1)
	x_test_n_r = x_test_n.reshape(-1, 28, 28, 1)

	#OnHotEncoder sur label
	y_train_c = to_categorical(y_train, num_classes = 10)
	y_val_c = to_categorical(y_val, num_classes = 10)
	y_test_c = to_categorical(y_test, num_classes = 10)

	#convert in array (mandatory with tensorflow)
	y_train_a = np.array(y_train_c)
	y_val_a = np.array(y_val_c)
	y_test_a = np.array(y_test_c)

	return (x_train_n_r, x_test_n_r, x_val_n_r), (y_train_a, y_test_a, y_val_a)
	

def create_model():
	#-------MODEL------
	model = tf.keras.Sequential()

	###CNN 
	#16 filtres avec une fenêtre de 3x3
	#Ici que des chiffres en NB et 28x28 donc suffit largement
	#Mais si pas assez précis car images de dim plus gdes / RGB => diminuer fenetre et aug filtres
	model.add(Conv2D(16, (3, 3), input_shape=(28, 28, 1) ))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(16, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())  # convertir nos 3D feature maps en 1D feature vectors

	model.add(Dense(16))
	model.add(Activation('relu'))

	#ATTENTION !! Dernière couche doit contenir le même nb de neurones que le nb de Labels !
	model.add(Dense(10)) #chiffres de 0 à 9 => 10 labels
	model.add(Activation('sigmoid'))
	return model

def train_model(model, x_train, y_train, x_val_r, y_val_a):
	#------TRAINNIG------
	#optimizer ?  
	#commence par compiler le "reseau"
	#params sur site keras ; metrics (metrique de calcul, manhattan,...), loss (calcul du coût), optimizer (copmliqué...)
	model.compile(
		optimizer='adam', 
		loss='sparse_categorical_crossentropy', 
		metrics=['accuracy']
		)

	print(model.summary())

	#entrainement du reseau ; ici rapide mais peut être très long si nos ccouches conv 
	#possèdes bcp de filtres et nos fenetres sont petites...


	model.fit(x_train, y_train, 
		batch_size=32, 
		epochs=10, 
		validation_data=(x_val_r, y_val_a),
		)


def evaluate_model(model, x_test, y_test):
	val_loss, val_accuracy = model.evaluate(x_test, y_test)
	print("Cout de : ", val_loss) #environ 0.06
	print("Precision de : ", val_accuracy) #environ 0.98


def predict_result(model, index_test_set, features, label):
	label_true = label[index_test_set]
	feature = features[index_test_set].reshape(1, 28, 28, 1)
	predict = model.predict(feature)
	label_pred = np.argmax(predict)
	print("Le vrai chiffre est (OneHotEncoder) : ", label_true)
	print("Modèle prédit : ", (predict), "donc : " ,label_pred)
	plt.imshow(features[index_test_set])
	plt.show()

def predict_result_extern_data(model, image, label):
	label_true = label
	image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
	#image = np.dot(image[...,:3], [0.299, 0.587, 0.144])
	image = image.reshape(1, 28, 28, 1)
	image = utils.normalize(image, axis=1)
	image = np.array(image)
	predict = model.predict(image)
	#label_pred = np.argmax(model.predict(y_test_a[index_test_set]))
	label_pred = np.argmax(predict)
	print("Data extern label : ", label_true)
	print("Modèle prédit : ", (predict), "donc : " ,label_pred)
	plt.imshow(image)
	plt.show()

def  show_results(img_ext):
	img_ext = cv2.resize(img_ext, (28, 28), interpolation=cv2.INTER_AREA)
	#image = np.dot(image[...,:3], [0.299, 0.587, 0.144])
	#img_ext = img_ext.reshape(1, 28, 28, 1)
	img_ext = utils.normalize(img_ext, axis=1)
	img_ext = np.array(img_ext)
	plt.figure()
	plt.subplot(2, 3, 1)
	plt.title("MNIST : prediction 7 (99%)", color='blue', size=18)
	plt.imshow(x_test_n_r[0], cmap='plasma')
	plt.subplot(2, 3, 2)
	plt.title("MNIST : prediction 5 (95%)", color='blue', size=18)
	plt.imshow(x_test_n_r[52], cmap='plasma')
	plt.subplot(2, 3, 3)
	plt.title("MNIST : prediction 6 (91%)", color='blue', size=18)
	plt.imshow(x_test_n_r[54], cmap='plasma')
	plt.subplot(2, 3, 4)
	plt.title("MNIST : prediction 4 (96%)", color='blue', size=18)
	plt.imshow(x_test_n_r[56], cmap='plasma')
	plt.subplot(2, 3, 5)
	plt.title("Extern data : prediction 8 (95%)", color='blue', size=18)
	plt.imshow(img_ext, cmap='plasma')
	plt.show()




###______________MAIN___________### -----------------------------------------------------------------------------

#charger dataset => images 28x28 codées digits de 0à9 
#extraire Features et Labels de test et de training
(x_dev, y_dev), (x_test, y_test) = mnist.load_data()

#create train set and test set
(x_train_n_r, x_test_n_r, x_val_n_r), (y_train_a, y_test_a, y_val_a) = preprocessing_data_mnist(x_dev, y_dev, x_test, y_test)  

"""
#if need to create and train model (10min with i5 9th core) ! 
#create model
model = create_model()
#train model using validation set to set up hyper-parameters
train_model(model, x_train_n_r, y_train_a, x_val_n_r, y_val_a)
#evaluate model
evaluate_model(model, x_test_n_r, y_test_a)
"""

#load model
model = models.load_model("cnn_mnist.h5")

print(model.summary())

predict_result(model, 0, x_test_n_r, y_test_a)



"""
#Extern datas (handwritten number of my friend)
image_mainguy = plt.imread("image_8_mainguy.jpg")
image_mainguy = cv2.imread("image_8_mainguy.jpg" ,cv2.IMREAD_GRAYSCALE)

predict_result_extern_data(model, image_mainguy, 8)

show_results(image_mainguy)

"""

