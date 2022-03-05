from colorize import *
from models import * 
from keras.layers import InputLayer, Conv2D, UpSampling2D, Dense, LeakyReLU, Dropout

encoder = Block([
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

decoder = Block([
	InputLayer(input_shape=(None, None, 256)), 
	Conv2D(128, (3, 3), activation="relu", padding="same"),
	UpSampling2D((2, 2)),
	Conv2D(64, (3, 3), activation="relu", padding="same"),
	UpSampling2D((2, 2)),
	Conv2D(32, (3, 3), activation="relu", padding="same"),
	Conv2D(2, (3, 3), activation="tanh", padding="same"),
	UpSampling2D((2, 2))
], name='decoder') 

classifier = VGG19()

model = VAE(encoder, decoder, classifier)
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.0001), 
    loss=tf.keras.losses.MeanSquaredError()
    )


discriminator = tf.keras.models.Sequential([
	InputLayer(input_shape=(None, None, 2)), 
	Conv2D(128, (3, 3), activation='relu', padding='same'),
	Conv2D(128, (3, 3), activation='relu', padding='same', strides=(2, 2)),
	Dropout(0.3),
	LeakyReLU(),
	Conv2D(256, (3, 3), activation='relu', padding='same'),
	Dropout(0.3),
	LeakyReLU(), 
	Conv2D(256, (3, 3), activation='relu', padding='same', strides=(2, 2)),
	Dense(1)
	])

gan = GAN(model, discriminator)


sample = Folder('tmp_val')
val = Folder('tmp_test', lenght=len(sample.directlist))
gan.fit(sample, val)


