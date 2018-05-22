import numpy as np
import tensorflow as tf
import random

class Loader():

	def __init__(self,batch_size=64):

		(train_x,train_y), (test_x,test_y) = tf.keras.datasets.cifar100.load_data(label_mode='fine')


		# Normalize data
		train_x = (train_x - np.mean(train_x))/((np.max(train_x) - np.min(train_x)))
		test_x = (test_x - np.mean(test_x))/((np.max(test_x) - np.min(test_x)))

		self.batch_size = batch_size

		self.train_x = train_x
		self.train_y = train_y
		self.test_x = test_x
		self.test_y = test_y

	def get_dataset(self):

		dataset =  tf.data.Dataset.from_tensor_slices((self.train_x[0:400],np.ravel(self.train_y[0:400]))).prefetch(buffer_size=1000).batch(64).repeat()
		iterator = dataset.make_one_shot_iterator()

		return iterator


	def get_train_batch(self,normalized=False):

		while True:
			# Creates a generator with a random mini batch
			sample = random.sample(list(np.arange(self.train_x.shape[0])),self.batch_size)			

			yield self.train_x[sample], np.ravel(self.train_y[sample])
