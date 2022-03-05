import tensorflow as tf 
from colorize import Folder, deprocessed_img, write
from PIL import Image
from models import * 
from tensorflow.keras.layers import InputLayer, Conv2D, UpSampling2D, Dense, LeakyReLU, Dropout
import matplotlib.pyplot as plt 

test = Folder('tmp_test', lenght=100)

TheWorst = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)] 
TheBest = [(0, 10), (0, 10), (0, 10), (0, 10), (0, 10)]

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
	Conv2D(384, (3, 3), activation='relu', padding='same')
], name='encoder')


decoder = Block([
	InputLayer(input_shape=(None, None,1356)), 
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
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7), 
    loss=tf.keras.losses.MeanAbsoluteError()
    )
model.load_weights('model2/weights')

losses = [] 

metric = tf.keras.losses.MeanAbsoluteError()
count = 0


for x, y, s in test:
    
    predict = model.predict(x)
    loss = tf.reduce_mean(metric(predict, y))
    losses += [loss]

    TheBest += [(deprocessed_img(predict, x, (256, 256)), loss)]
    # TheBest.sort(key=lambda x: x[1])
    TheBest = TheBest[:-1]

    TheWorst += [(deprocessed_img(predict, x, (256, 256)), loss)]
    # TheWorst.sort(key=lambda x: x[1])
    TheWorst = TheWorst[1:]

    write(f'count: {count}')
    count += 1

name = 'VAE'

for img, loss in TheWorst:
    plt.imshow(img)
    plt.show() 

for img, loss in TheBest:
    try:
        plt.imshow(img)
        plt.show()
    except: pass 


plt.plot(list(range(len(losses))), losses)
plt.show()