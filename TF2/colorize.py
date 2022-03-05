import tensorflow as tf 
from PIL import Image 
import matplotlib.pyplot as plt 
from skimage.color import rgb2lab, lab2rgb
import numpy as np

import sys, os, random, time, os

"""
дата: 06.02.2022
"""


class CompileError(Exception): pass 
class WorkError(Exception): pass 
class PathError(Exception): pass 
class UseError(Exception): pass 

class FusinoLayer(tf.Module):
	"""
	## FusinoLayer 
	- - - 
	функтор, который получает на вход список из двух выходов других слоев, 
	один из которых вектор, а второй тензор.
	Первыйм в списке должен быть вектор, а второым тензор

	inpurs:list
	"""
	def __init__(self):
		super().__init__()

	@tf.function
	def __call__(self, inputs: list) -> tf.Tensor:
		imgs, embs = inputs
		reshaped_shape = imgs.shape[:3].concatenate(embs.shape[1])
		embs = tf.repeat(embs, imgs.shape[1] * imgs.shape[2])
		embs = tf.reshape(embs, reshaped_shape)
		return tf.concat([imgs, embs], axis=3)

class Folder:
	"""
	## date_images 
	- - - 
	итерируемый объект, который при каждой итерации возвращает объект неизменное изображения и 
	цветовцю характеристику из формата LAB. Данный класс выбирает случайным образом изображение из указанной директории 

	path: str -- строка до директории с изображеними
	"""
	def __init__(self, path:str, lenght:int = 0):
		if type(path) != str: raise PathError('тип path должен быть str')
		self.path = path 
		self.directlist = os.listdir(self.path)

		if not lenght:
			self.lenght = len(os.listdir(self.path))
		else:
			self.lenght = lenght

		self.count = 0
	
	def __iter__(self):
		return self

	def setSeed(self, seed:int):
		random.seed(seed)

	def __next__(self):
		if self.count <= self.lenght:
			img_l = self.directlist[random.randint(0, len(self.directlist)-1)]
			img = Image.open(f'{self.path}/{img_l}')
			self.count += 1
			img = np.array(img)
			return self.get_y(img)
		else: 
			raise StopIteration

	def get_y(self, img):
		try:
			image = np.array(img, dtype=float)
			size = image.shape

			lab = rgb2lab(1.0/255*image)
			X, Y = lab[:,:,0], lab[:,:,1:]

			Y /= 128    # нормируем выходные значение в диапазон от -1 до 1
			X = X.reshape(1, size[0], size[1], 1)
			Y = Y.reshape(1, size[0], size[1], 2)
	
			return X, Y, len(self.directlist)

		except:
			return self.__next__()

class AutoEncoder(tf.keras.Model):
	"""
	# AutoEncoder 
	- - -
	модель автоинкодера. Прежед чем использовать объект данного класса, необходимо его скомпелировать. Для обучения необходим итерируемый объект или генероатор, который возвращает изображение в неизменном виде и
	цветовую характеристику данного изображения из формата [LAB](https://ru.wikipedia.org/wiki/LAB). 

	Сеть принимате строго одно изображение и одну цветовую характеристику за однин проход.  
	
	!!!Обратите внимание, количество входных каналов декодера должно быть равно количеству каналлов инкодера + количесвто каналов классификатора 

	"""
	def __init__(self, encoder:tf.keras.models.Sequential, 
				decoder:tf.keras.models.Sequential, 
				classifier:tf.keras.models.Sequential=None):
		super().__init__()
		self.encoder = encoder 
		self.decoder = decoder 
		self.classifier = classifier 
		self.fl_compile = False
		self.fl_train = False
		self.history = {
			'loss': [],
			'time': []
		}

	def save(self, name='AutoEncoder', history=None):
		if history is None:
			history = self.history

		model = saved_network(self.encoder, self.decoder, self.classifier, history)

		tf.saved_model.save(model, name, signatures=model.__call__.get_concrete_function(tf.TensorSpec(shape=[1, 256, 256, 1], dtype=tf.float32, name='net')))



	def compile(self, loss:tf.keras.losses=tf.keras.losses.MeanSquaredError(), 
				optimizer:tf.keras.optimizers=tf.keras.optimizers.Adam(learning_rate=0.0001)):
		"""
		Метод компиляции, который должен быть вызван один раз 
		По умолчанию:
			loss: Минимум квадрата ошибки 
			optimaizer: Adam

		loss: функция потерь. Не должна включать в себя функции пакета numpy и в приоритете функции пакета tensorflow
		optimizer: оптимизатор. Должен быть взят из ветки tensorflow.optimiaers
		"""
		if not self.fl_compile:
			self.loss = loss 
			self.optimizer = optimizer
			self.fl_compile = True
		else:
			raise CompileError('Повторное компелирование модели запрещено')

	def fit(self, generator, epochs=1):
		"""
		Метод обучения, который обучает модель epochs раз. Генератор должен возвращать изображение в неизменном виде и 
		цветовую характеристику этого изображения в формате LAB.

		generator: Генератор 
		epochs: количество эпох
		"""
		self.fl_train = True
		count = 0
		imgn = 0
		
		for i in range(epochs):
			for x, y, size in generator:
				t = time.time()
			
				loss = self.train(x, y)
				proc = round((i/epochs)*30)
				write(f'loss: {float(loss):^3.5} | {count}')
				if not count%2000:
					self.save()		

					
				imgn += 1
				count += 1

			t = time.time() - t
			self.history['time'] += [t]
			self.history['loss'] += [loss]

		else:
			self.save()
			

	@tf.function
	def call(self, x):
		if not self.fl_compile:
			raise CompileError('Перед использованием, модель должна быть скомпелирована')

		vector = self.encoder(x)
		if not (self.classifier is None):
			class_img = self.classifier(x)
			vector = FusinoLayer()([vector, class_img])
			del class_img

		return self.decoder(vector)

	@tf.function
	def train(self, x, y):
		with tf.GradientTape() as tape:
			predict = self.call(x)
			loss_f = self.loss(y, predict)

		grads = tape.gradient(loss_f, self.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
		return loss_f

	@tf.function
	def __call__(self, img):
		return self.call(img)
		

def deprocessed_img(output, grayimage, size):
	"""
	# Deoricessed_img
	- - - 
	## Переводит lab в rgb
	output: цветовая характеристика изображения 

	grayimage: черно-белое изображени 

	size; размер исходного изображения 
	"""

	output *= 128
	min_vals, max_vals = -128, 127
	ab = np.clip(output[0], min_vals, max_vals)

	cur = np.zeros((size[0], size[1], 3))
	cur[:,:,0] = np.clip(grayimage[0][:,:,0], 0, 100)
	cur[:,:,1:] = ab
	return lab2rgb(cur)


class saved_network(tf.keras.Model):
	def __init__(self, encoder, decoder, classifier, history,  name=None):
		super().__init__(name=name)
		self.history = history
		self.encoder = encoder
		self.decoder = decoder 
		self.classifier = classifier

	@tf.function
	def __call__(self, x):
		vector = self.encoder(x)

		if self.classifier is None:
			return self.decoder(vector)

		class_img = self.classifier(x)
		vector = FusinoLayer()([vector, class_img])

		return self.decoder(vector)

class GAN(tf.keras.Model):
	def __init__(self, generator, discriminator, name=None):
		super().__init__(name=name)
		self.generator = generator 
		self.discriminator = discriminator
		self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
		self.history = {
			'gen_loss': [], 
			'disc_loss': [], 
			'time': [],
			'validation_loss': [], 
			'meansqr':[]
					}


	def fit(self, sample, validate, optimizer_disc=tf.keras.optimizers.Adam(learning_rate=3e-4), 
						optimizer_gen=tf.keras.optimizers.Adam(learning_rate=1e-7), 
						epochs=10):
		count = 0
		
		for x, y, s in sample:
			t = time.time()
			gen_loss, disc_loss , generate = self.learnstep(x, y, optimizer_disc, optimizer_gen)
				
			t = time.time() - t 

			self.history['time'] += [t]
			self.history['gen_loss'] += [gen_loss]
			self.history['disc_loss'] += [disc_loss]
			self.history['meansqr'] += [tf.reduce_mean(tf.square(y - generate))]

			if not count%2000:
				self.generator.save('GAN')

			write(f'[{count:^4}] gen_loss: {float(tf.reduce_mean(gen_loss))} | disc_loss: {disc_loss} | mean_sqr: {tf.reduce_mean(tf.square(y - generate))}')
			count += 1
				
		self.generator.save(name='GAN', history=self.history)


	def show_plot(self):
		plt.clf() 
		plt.plot(self.history['gen_loss'])
		plt.plot(self.history['disc_loss'])
		plt.plot(self.history['validation_loss'])
		plt.draw()
		plt.gcf().canvas.flush_events()


	def get_val_loss(self, x, y):
		"""
		Возвращает картеж (ошибка генератора, ошбка дискриминатора)
		"""
		with tf.GradientTape() as tape:
			generate = self.generator(x)

			fake = self.discriminator(generate)
			real = self.discriminator(y) 

		return self.generator_loss(fake), self.diskriminator_loss(real, fake)		

	@tf.function
	def learnstep(self, x, y, optimizer_disc, optimizer_gen):
		with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
			generate = self.generator(x)

			real = self.discriminator(y)
			fake = self.discriminator(generate)

			gen_loss = self.generator_loss(fake)
			disc_loss = self.diskriminator_loss(real, fake)

		gradient_gen = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
		gradient_disc = dis_tape.gradient(disc_loss, self.discriminator.trainable_variables)

		optimizer_gen.apply_gradients(zip(gradient_gen, self.generator.trainable_variables))
		optimizer_disc.apply_gradients(zip(gradient_disc, self.discriminator.trainable_variables))
		
		return gen_loss, disc_loss, generate

	@tf.function
	def generator_loss(self, output):
		loss = self.cross_entropy(tf.ones_like(output), output) 
		return loss

	@tf.function
	def diskriminator_loss(self, real, fake):
		real_loss = self.cross_entropy(tf.ones_like(real), real)
		fake_loss = self.cross_entropy(tf.zeros_like(fake), fake)
		return real_loss + fake_loss
				
	@tf.function
	def __call__(self, x):
		return self.generator(x)

def write(str, file=sys.stdout):
	file.write('\r'+str + ' ' * 7)
	file.flush()

class VGG19(tf.Module):
	def __init__(self):
		self.vgg19 = tf.keras.applications.VGG19(weights='imagenet')
		self.vgg19.trainable = False

	@tf.function
	def __call__(self, img):
		img = tf.image.resize_with_crop_or_pad(img[0], 224, 224)
		img = tf.concat([img, img, img], axis=2)
		img = tf.expand_dims(img, axis=0)
		return self.vgg19(img)

if __name__ == '__main__':
	raise UseError('данный файл должен использоваться как модуль')