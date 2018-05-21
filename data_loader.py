import numpy as np
import tensorflow as tf
import random

class Loader():

	def __init__(self,batch_size=32):

		(train_x,train_y), (test_x,test_y) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

		self.batch_size = batch_size

		self.train_x = train_x
		self.train_y = train_y
		self.test_x = test_x
		self.test_y = test_y


	def get_batch(self):
		while True:
			sample = random.sample(list(np.arange(self.train_x.shape[0])),self.batch_size)
			yield self.train_x[sample], self.train_y[sample]
