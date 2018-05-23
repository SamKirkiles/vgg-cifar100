import tensorflow as tf
import numpy as np
from data_loader import Loader



class VGG:
 
	def __init__(self):

		loader = Loader()
		iterator = loader.get_dataset()


		def build_model():

			with tf.device("/device:GPU:0"):
				x_loaded,y_loaded = iterator.get_next()

				x = tf.placeholder_with_default(x_loaded,(None,32,32,3),name="x_placeholder")
				y = tf.placeholder_with_default(y_loaded,(None),name="y_placeholder")

				training = tf.placeholder_with_default(True,name="training_bool",shape=())

				#Layer1 - 64 channels
				conv1 = tf.layers.conv2d(x, filters=64,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
				# Layer2 - 64 channels
				conv2 = tf.layers.conv2d(conv1, filters=64,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
				pool2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='SAME')

				#Layer 3 - 128 channels
				conv3 = tf.layers.conv2d(pool2, filters=128,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
				# Layer 4 - 128 channels
				conv4 = tf.layers.conv2d(conv3, filters=128,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
				pool4 = tf.layers.max_pooling2d(conv4, (2,2), (2,2), padding='SAME')

				#Layer 5 - 256 channels
				conv5 = tf.layers.conv2d(pool4, filters=256,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
				# Layer 6 - 256 channels
				conv6 = tf.layers.conv2d(conv5, filters=256,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
				# Layer 7 - 256 channels
				conv7 = tf.layers.conv2d(conv6, filters=256,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())

				pool7 = tf.layers.max_pooling2d(conv7, (2,2), (2,2), padding='SAME')

				# Layer 8 - 512 channels
				conv8 = tf.layers.conv2d(pool7, filters=512,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())

				# Layer 9 - 512 channels
				conv9 = tf.layers.conv2d(conv8, filters=512,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
				# Layer 10 - 512 channels
				conv10 = tf.layers.conv2d(conv9, filters=512,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())

				pool10 = tf.layers.max_pooling2d(conv10, (2,2), (2,2), padding='SAME')

				# Layer 11 - 512 channels
				conv11 = tf.layers.conv2d(pool10, filters=512,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
				# Layer 12 - 512 channels
				conv12 = tf.layers.conv2d(conv11, filters=512,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
				# Layer 13 - 512 channels
				conv13 = tf.layers.conv2d(conv12, filters=512,kernel_size=(3,3),padding='SAME',activation=tf.nn.relu,
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())

				pool13 = tf.layers.max_pooling2d(conv13, (2,2), (2,2), padding='SAME')


				flattened = tf.contrib.layers.flatten(pool13)

				dense14 = tf.layers.dense(inputs=flattened, units=4096,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
				dense15 = tf.layers.dense(inputs=dense14, units=4096,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
				dense16 = tf.layers.dense(inputs=dense15, units=100,activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer())


				scaled_logits = -tf.log(dense16)
				outputs = tf.argmax(tf.nn.softmax(scaled_logits),axis=0)


				softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=dense16,name="softmax")

				loss = tf.reduce_mean(softmax)

				optimize = tf.train.AdamOptimizer(0.1).minimize(loss)

			self.loss = loss
			self.x_placeholder = x
			self.y_placeholder = y
			self.training = training
			self.outputs = outputs
			self.softmax = softmax
			self.optimize = optimize

			tf.summary.scalar("Loss", loss)

		build_model()


	def train(self,restore=False):

		saver = tf.train.Saver()


		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
			try:

				run_id = np.random.randint(0,1e7)

				train_writer = tf.summary.FileWriter(logdir="logs/" + str(run_id), graph=sess.graph)

				if restore:
					saver.restore(sess, tf.train.latest_checkpoint('./saves'))
				else:
					sess.run(tf.global_variables_initializer())

				counter = 0

				train_loader = Loader()
				val_x,val_y = sess.run(train_loader.get_dataset(train=False).get_next())

				while True:

						counter += 1
						merge = tf.summary.merge_all()

						_, summary = sess.run([self.optimize,merge],feed_dict={})

						train_writer.add_summary(summary,counter)

						if counter%1000 == 0:

							# Check validation accuracy
							outputs = sess.run([self.outputs],feed_dict={self.x_placeholder:val_x,self.y_placeholder:val_y,self.training:False})
							accuracy = np.mean(outputs == val_y)

							accuracy_summary = tf.Summary(value=[tf.Summary.Value(tag='Accuracy',simple_value=accuracy)])
							train_writer.add_summary(accuracy_summary,counter)

							# Save model
							print("Periodically saving model...")
							save_path = saver.save(sess, "./saves/model.ckpt")

			except KeyboardInterrupt:
				print("Interupted... saving model.")
			
			save_path = saver.save(sess, "./saves/model.ckpt")


	def test(self,inputs,labels,restore=True):

		with tf.Session() as sess:

			if restore:
				saver.restore(sess, tf.train.latest_checkpoint('./saves'))
			else:
				sess.run(tf.global_variables_initializer())


			output = sess.run([self.output_distribution], feed_dict={self.x_placeholder:inputs})






