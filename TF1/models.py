import tensorflow as tf 
from PIL import Image 
import matplotlib.pyplot as plt 
from skimage.color import rgb2lab, lab2rgb
import numpy as np
import sys, os, random, time, os

class CompileError(Exception): pass 
class WorkError(Exception): pass 
class PathError(Exception): pass 
class UseError(Exception): pass 

class Block(tf.keras.layers.Layer):
	def __init__(self, layers:list, name=None) -> None:
		super(Block, self).__init__(name=name)
		self.layers = layers

	@tf.function
	def call(self, input):
		x = input
		for layer in self.layers:
			x = layer(x)
			
		return x        

class FusinoLayer(tf.keras.layers.Layer):
	def __init__(self, name=None):
		super(FusinoLayer, self).__init__(name=name)

	def call(self, inputs: list) -> tf.Tensor:
		imgs, embs = inputs
		imgs = tf.expand_dims(imgs[0], axis=0)
		reshaped_shape = imgs.shape[:3].concatenate(embs.shape[1])
		embs = tf.repeat(embs, imgs.shape[1] * imgs.shape[2])
		embs = tf.reshape(embs, reshaped_shape)
		return tf.concat([imgs, embs], axis=3)


class VGG19(tf.keras.layers.Layer):
	def __init__(self, name=None):	
	
		super(VGG19, self).__init__(name=name)
		self.vgg19 = tf.keras.applications.VGG19(weights='imagenet')
		self.vgg19.trainable = False

	def call(self, img):
		img = tf.image.resize_with_crop_or_pad(img[0], 224, 224)
		img = tf.concat([img, img, img], axis=2)
		img = tf.expand_dims(img, axis=0)
		return self.vgg19(img)


class VAE(tf.keras.Model):
	def __init__(
        self, 
        Encoder:tf.keras.layers.Layer, 
        Decoder:tf.keras.layers.Layer, 
        Classifier:tf.keras.layers.Layer = None, 
        name = None):

		super(VAE, self).__init__(name=name)
		self.fusion = FusinoLayer()
		self.encoder = Encoder
		self.decoder = Decoder
		self.classifier = Classifier
	
	@tf.function
	def call(self, x):
		vector = self.encoder(x)

		if self.classifier is None:
			return self.decoder(vector)

		class_img = self.classifier(x)
		vector = self.fusion([vector, class_img])
		
		return self.decoder(vector)


class Folder:
	def __init__(self, path:str, lenght:int = 0):
		self.path = path 
		self.directlist = os.listdir(self.path)
		self.lenght = lenght if lenght else len(os.listdir(self.path))
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

			
			Y /= 128 
			X = X.reshape(1, size[0], size[1], 1)
			Y = Y.reshape(1, size[0], size[1], 2)
	
			return X, Y, len(self.directlist)

		except:
			return self.__next__()


def write(str, file=sys.stdout):
	file.write('\r'+str + ' ' * 7)
	file.flush()


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