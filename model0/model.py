from models import * 
from tensorflow.keras.layers import InputLayer, Conv2D, UpSampling2D

encoder = tf.keras.Sequential([
	InputLayer(input_shape=(None, None, 1)), 
	Conv2D(64, (3, 3), activation='relu', padding='same'),
	Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),
	Conv2D(128, (3, 3), activation='relu', padding='same'),
	Conv2D(128, (3, 3), activation='relu', padding='same', strides=2),
	Conv2D(256, (3, 3), activation='relu', padding='same'),
	Conv2D(256, (3, 3), activation='relu', padding='same', strides=2),
	Conv2D(512, (3, 3), activation='relu', padding='same'),
	Conv2D(512, (3, 3), activation='relu', padding='same'),	
	Conv2D(384, (3, 3), activation='relu', padding='same')
], name='encoder')


decoder = tf.keras.Sequential([
	InputLayer(input_shape=(None, None,1356)), 
	Conv2D(128, (3, 3), activation="relu", padding="same"),
	UpSampling2D((2, 2)),
	Conv2D(64, (3, 3), activation="relu", padding="same"),
	UpSampling2D((2, 2)),
	Conv2D(32, (3, 3), activation="relu", padding="same"),
	Conv2D(2, (3, 3), activation="tanh", padding="same"),
	UpSampling2D((2, 2))
], name='decoder') 



model = VAE(encoder, decoder)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7), 
    loss=tf.keras.losses.MeanAbsoluteError()
    )


model.load_weights('model2/weights')
