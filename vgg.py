import tensorflow as tf
import numpy as np

class VGG:
 
	def __init__(self):

		def build_model():

			x = tf.placeholder(dtype=tf.float32,shape=(None,32,32,3),name="inputs")
			y = tf.placeholder(dtype=tf.float32,shape=(None,),name="labels")

			#Layer1 - 64 channels
			conv1 = tf.layers.conv2d(x, filters=64,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
				use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
			# Layer2 - 64 channels
			conv2 = tf.layers.conv2d(x, filters=64,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
				use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
			pool2 = tf.nn.maxpool(conv2,ksize=2,strides=2,padding="SAME")

			#Layer 3 - 128 channels
			conv3 = tf.layers.conv2d(pool2, filters=128,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
				use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
			# Layer 4 - 128 channels
			conv4 = tf.layers.conv2d(conv3, filters=128,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
				use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
			pool4 = tf.nn.maxpool(conv4,ksize=2,strides=2,padding="SAME")

			#Layer 5 - 256 channels
			conv5 = tf.layers.conv2d(pool4, filters=256,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
				use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
			# Layer 6 - 256 channels
			conv6 = tf.layers.conv2d(conv5, filters=256,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
				use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
			# Layer 7 - 256 channels
			conv7 = tf.layers.conv2d(conv6, filters=256,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
				use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())

			pool7 = tf.nn.maxpool(conv7,ksize=2,strides=2,padding="SAME")

			# Layer 8 - 512 channels
			conv8 = tf.layers.conv2d(pool7, filters=512,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
				use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())

			# Layer 9 - 512 channels
			conv9 = tf.layers.conv2d(conv8, filters=512,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
				use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
			# Layer 10 - 512 channels
			conv10 = tf.layers.conv2d(conv9, filters=512,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
				use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())

			pool10 = tf.nn.maxpool(conv10,ksize=2,strides=2,padding="SAME")

			# Layer 11 - 512 channels
			conv11 = tf.layers.conv2d(pool10, filters=512,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
				use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
			# Layer 12 - 512 channels
			conv12 = tf.layers.conv2d(conv11, filters=512,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
				use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
			# Layer 13 - 512 channels
			conv13 = tf.layers.conv2d(conv12, filters=512,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
				use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())

			pool13 = tf.nn.maxpool(conv13,ksize=2,strides=2,padding="SAME")

			flattened = tf.contrib.layers.flatten(pool13)

			dense14 = tf.layers.dense(flattened, units=4096,activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.xavier_initializer())
			dense15 = tf.layers.dense(dense14, units=4096,activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.xavier_initializer())
			dense16 = tf.layers.dense(dense15, units=1000,activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.xavier_initializer())



