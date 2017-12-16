from keras.models import Model,Sequential
from keras.models import load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, ZeroPadding2D
from keras.utils import np_utils, plot_model # utilities for one-hot encoding of ground truth values
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pprint
from sklearn.metrics import confusion_matrix

#Where to save the weights and model
WEIGHTS_FILEPATH = 'cnn.best.weights.hdf5'
OUTPUT_PATH = 'cnn.model.hdf5'


#Input folders
INPUT_TRAIN_FOLDER = './images/train/'
INPUT_VALID_FOLDER = './images/valid/'
INPUT_TEST_FOLDER = './images/test/'

#Testing
#Load the weights file
TESTING_WEIGHTS_PATH = 'cnn.best.weights.hdf5'
#Destination of predictions
TESTING_RESULTS_PATH = 'test-s9.csv'



def buildmodel():

	'''
	3. This convolution layer uses 64 10X10 convolutions with a 1X1 stride and 5X5 padding to further convolve the features folowed by a Rectified Linear Unit (ReLU) layer to set all negative elements to zero.
	4. The convolved features then go into the maximum pooling layer. The pooling layer cal- culates the maximum value of the feature over a region of the image so we can use the features for classification. This max pooling layer has a filter size of 3X3.


	'''
	#batch_size = 10 #tbd


	DATA_SIZE = (512,512, 1)
	STRIDE_SIZE = (1,1)
	PADDING_SIZE = (5,5)
	POOLING_SIZE = (3,3)
	DROPOUT_PROB = 0.1


	#5x5 padding for data
	zeropadding_1 = ZeroPadding2D(padding = PADDING_SIZE, input_shape = DATA_SIZE)


	#First Convolutional Layer
	#CONV64, 10x10, 1x1 strides, relu
	conv_2d_layer_1 = Conv2D(filters = 64, kernel_size = 10, padding = 'valid', strides = STRIDE_SIZE, activation = 'relu')

	#3x3 Pooling
	pool_1 = MaxPooling2D(pool_size = POOLING_SIZE)

	#10% dropout
	drop_1 = Dropout(0.1)


	#Second convolutional layer
	zeropadding_2 = ZeroPadding2D(padding = PADDING_SIZE, input_shape = DATA_SIZE)

	#CONV192, 5x5, 1x1 strides, relu
	conv_2d_layer_2 = Conv2D(filters = 192, kernel_size = 5, padding = 'valid', strides = STRIDE_SIZE, activation = 'relu')

	#2x2 Pooling
	pool_2 = MaxPooling2D(pool_size = (2,2))

	#10% dropout
	drop_2 = Dropout(0.1)


	#Convolutional layers 3 to 7
	#CONV384, 5x5, 1x1 strides, relu, NO PADDING
	conv_2d_layer_3 = Conv2D(filters = 384, kernel_size = 5, padding = 'valid', strides = STRIDE_SIZE, activation = 'relu')
	conv_2d_layer_4 = Conv2D(filters = 256, kernel_size = 3, padding = 'valid', strides = STRIDE_SIZE, activation = 'relu')
	conv_2d_layer_5 = Conv2D(filters = 256, kernel_size = 3, padding = 'valid', strides = STRIDE_SIZE, activation = 'relu')
	conv_2d_layer_6 = Conv2D(filters = 256, kernel_size = 3, padding = 'valid', strides = STRIDE_SIZE, activation = 'relu')
	conv_2d_layer_7 = Conv2D(filters = 128, kernel_size = 3, padding = 'valid', strides = STRIDE_SIZE, activation = 'relu')

	pool_3 = MaxPooling2D(pool_size = (3,3))

	drop_3 = Dropout(0.5)

	flat = Flatten()
	fc_layer_1 = Dense(32, activation='relu')
	out = Dense(1, activation='sigmoid')
	
	#Fully connected layer with softmax
	#fc_layer = Dense(1, activation = 'softmax')

	model = Sequential()
	model.add(zeropadding_1)
	model.add(conv_2d_layer_1)
	model.add(pool_1)
	model.add(drop_1)

	model.add(zeropadding_2)
	model.add(conv_2d_layer_2)
	model.add(pool_2)
	model.add(drop_2)

	model.add(conv_2d_layer_3)
	model.add(conv_2d_layer_4)
	model.add(conv_2d_layer_5)
	model.add(conv_2d_layer_6)
	model.add(conv_2d_layer_7)
	model.add(pool_3)
	model.add(drop_3)

	model.add(flat)
	model.add(fc_layer_1)
	model.add(out)

	#model.add(fc_layer)

	model.compile(loss='binary_crossentropy', # using the cross-entropy loss function
				  optimizer='adam', # using the Adam optimiser
				  metrics=['accuracy']) # reporting the accuracy
	plot_model(model, to_file='model.png', show_shapes = True, show_layer_names = False)
	return model
	

def generate_images():
	#the split is performed beforehand
	train_datagen = ImageDataGenerator()
	valid_datagen = ImageDataGenerator()


	train_generator = train_datagen.flow_from_directory(
        INPUT_TRAIN_FOLDER,
        target_size=(512,512),
        batch_size=1,
        classes=['benign','cancer'],
        color_mode='grayscale',
        class_mode='binary')

	valid_generator = valid_datagen.flow_from_directory(
        INPUT_VALID_FOLDER,
        target_size=(512,512),
        batch_size=1,
        classes=['benign','cancer'],
        color_mode='grayscale',
        class_mode='binary')

	return train_generator, valid_generator


def train_model(model):

	#Model parameters
	batch_size = 25
	num_epochs = 100
	nb_train_samples = 330722
	nb_valid_samples = 163243

	checkpoint = ModelCheckpoint(WEIGHTS_FILEPATH, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	
	stopping = EarlyStopping(monitor='val_acc', min_delta=0.0007, patience=10, verbose=1, mode='auto')
	
	callbacks_list = [checkpoint, stopping]

	print("Training CNN")

	train_generator, valid_generator = generate_images()


	model.fit_generator( train_generator, epochs=num_epochs,
		validation_data= valid_generator,
		steps_per_epoch = nb_train_samples // batch_size,
		validation_steps = nb_valid_samples // batch_size,
		verbose=1,
		callbacks = callbacks_list
		)

	#save the model

	model.save(OUTPUT_PATH)

	print("Model trained and saved as {}".format(output_path))

#Use existing weights to predict model
def predict_model(model):

	nb_test_samples = 163243
	batch_size = 6

	test_datagen = ImageDataGenerator()
	test_generator = test_datagen.flow_from_directory(
        INPUT_TEST_FOLDER,
        target_size=(512,512),
        batch_size=batch_size,
        classes=['benign','cancer'],
        color_mode='grayscale',
        class_mode='binary',
        shuffle=False)	

	true_classes = test_generator.classes

	
	model.load_weights(TESTING_WEIGHTS_PATH)

	nb_steps = nb_test_samples//batch_size
	predicted_classes = model.predict_generator(test_generator, steps= nb_steps, verbose = 1)

	predicted_classes = list(map(lambda x: int(x), predicted_classes))

	#save the results
	print(predicted_classes)
	np.savetxt(TESTING_RESULTS_PATH, predicted_classes, delimiter=",", fmt='%d')
	print(confusion_matrix(true_classes, predicted_classes, labels=[0,1]))

	



def main():
	my_model = buildmodel()

	#train_model(my_model)
	#predict_model(my_model)


if __name__ == '__main__':
	main()

