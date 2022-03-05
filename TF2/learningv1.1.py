from keras.layers import InputLayer, Conv2D, UpSampling2D, Dense, LeakyReLU, Dropout
from colorize import *

encoder = tf.keras.models.Sequential([
	InputLayer(input_shape=(None, None, 1)), 
	Conv2D(64, (3, 3), activation='relu', padding='same'),
	Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),
	Conv2D(128, (3, 3), activation='relu', padding='same'),
	Conv2D(128, (3, 3), activation='relu', padding='same', strides=2),
	Conv2D(256, (3, 3), activation='relu', padding='same'),
	Conv2D(256, (3, 3), activation='relu', padding='same', strides=2),
	Conv2D(512, (3, 3), activation='relu', padding='same'),
	Conv2D(512, (3, 3), activation='relu', padding='same'),	
	Conv2D(256, (3, 3), activation='relu', padding='same')
], name='encoder')
 
decoder = tf.keras.models.Sequential([
	InputLayer(input_shape=(None, None, 256)), 
	Conv2D(128, (3, 3), activation="relu", padding="same"),
	UpSampling2D((2, 2)),
	Conv2D(64, (3, 3), activation="relu", padding="same"),
	UpSampling2D((2, 2)),
	Conv2D(32, (3, 3), activation="relu", padding="same"),
	Conv2D(2, (3, 3), activation="tanh", padding="same"),
	UpSampling2D((2, 2))
], name='decoder')


autoencoder = AutoEncoder(encoder, decoder)
autoencoder.compile()

for i in range(2):
	sample = Folder('tmp_val')
	val = Folder('tmp_test', lenght=len(sample.directlist))

	autoencoder.fit(sample)