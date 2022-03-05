from models import * 
from tensorflow.keras.layers import InputLayer, Conv2D, UpSampling2D


tmp = sys.stdout
sys.stdout = open('tmp1.txt', 'w')

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
	Conv2D(256, (3, 3), activation='relu', padding='same')
], name='encoder')


decoder = tf.keras.Sequential([
	InputLayer(input_shape=(None, None, 256)), 
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
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
    loss=tf.keras.losses.MeanSquaredError()
    )

def learn(epochs=1):
	count = 0
	model.save_weights('../model0/weights/weights')

	for i in range(epochs):
		date = Folder('../tmp_val')
		lenght = date.lenght*epochs

		for x, y, s in date:
			t = time.time()
			history = model.fit(x=x, y=y, epochs=1)
			count += 1
			proc = (50*count)//lenght
			t = time.time() - t
			t = round((lenght-count)*t)
			write('|'+proc*'=' + '>' + ' '*(50-proc)+'| ' + f'{(100*count)//lenght}% time: {t//(60*60)}:{(t%(60*60))//60}' + ' '*10, file=tmp)
			if not count%2000:
				model.save_weights('../model0/weights/weights')

	model.save_weights('../model0/weights/weights')

learn(5)