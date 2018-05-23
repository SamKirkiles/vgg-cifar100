import numpy as np
import tensorflow as tf
import random
import progressbar
import os.path
import matplotlib.pyplot as plt


class Loader():

	def __init__(self,batch_size=64):

		# Downloads the CIFAR 100 dataset and saves it as a TFRecords File 

		self.filenames = ["train.tfrecords","test.tfrecords"]

		found = True

		for file in self.filenames:
			if not os.path.isfile(file):
				found  = False
				print("TFRecords not found for file: " + file)

		if not found:
			(train_x,train_y), (test_x,test_y) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

			train_x = train_x.astype(np.float32)
			test_x = test_x.astype(np.float32)

			train_x = (train_x - np.mean(train_x))/(np.max(train_x)-np.min(train_x))
			test_x = (test_x - np.mean(test_x))/(np.max(test_x)-np.min(test_x))

			# Create validation
			self.val_x = test_x[0:test_x.shape[0]/2]
			self.val_y = test_y[0:test_y.shape[0]/2]
			
			self.create_tf_record(examples=train_x,labels=train_y,path="train.tfrecords")
			self.create_tf_record(examples=test_x,labels=test_y,path="test.tfrecords")

		self.batch_size = batch_size


	def get_dataset(self,train=True):

		if train:
			filenames = self.filenames[0]
			batch = self.batch_size
		else:
			filenames = self.filenames[1]
			batch = 1000

		dataset = tf.data.TFRecordDataset(filenames)

		# map this datset to our unserializing function
		dataset = dataset.map(self.parse_example,num_parallel_calls=8)
		dataset = dataset.repeat().shuffle(buffer_size=1000)
		dataset = dataset.batch(batch)
		dataset = dataset.prefetch(100)

		iterator = dataset.make_one_shot_iterator()

		return iterator

	def parse_example(self, serialized):

		features = {'image':(tf.FixedLenFeature((),tf.string,default_value="")),
					'label':(tf.FixedLenFeature((),tf.int64,default_value=0))}

		parsed = tf.parse_single_example(serialized=serialized, features=features)

		raw_image = parsed['image']

		image = tf.decode_raw(raw_image,tf.float32)

		return tf.reshape(image,[32,32,3]), parsed['label']

	def create_tf_record(self,examples,labels,path):

		# Takes training examples and labels to save in a .tfrecord file at the given path

		with tf.python_io.TFRecordWriter(path) as writer:

			# Make examples into serialized string
			# we can just save all images 

			# This is just a progress bar
			print("Writing " + path + "...")

			widgets = [progressbar.Percentage(), progressbar.Bar()]
			bar = progressbar.ProgressBar(widgets=widgets, max_value=examples.shape[0]).start()

			# Loop through all images
			for i in range(examples.shape[0]):
				
				# turn image into bytes and get our label
				img = examples[i].tostring()
				label = labels[i]

				# Now we need an example which is made up of a feature

				features = tf.train.Features(
					feature={
						'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
						'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
					}
				) 

				example = tf.train.Example(features=features)

				serialized = example.SerializeToString()

				writer.write(serialized)

				# Update the progressbar
				bar.update(i + 1)


			bar.finish()
