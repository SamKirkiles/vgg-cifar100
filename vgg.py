import tensorflow as tf
import numpy as np

class VGG:
 
	def __init__(self):

		loader = Loader()
		iterator = loader.get_dataset()


		def build_model():

			with tf.device("/cpu:0"):

				x,y = iterator.get_next()

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
				output_distribution = tf.nn.softmax(scaled_logits)

				softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=dense16,name="softmax")

				loss = tf.reduce_mean(softmax)

				optimize = tf.train.AdamOptimizer().minimize(loss)


			self.loss = loss
			self.output_distribution = output_distribution
			self.softmax = softmax
			self.optimize = optimize

			with tf.device("/cpu:0"):
				tf.summary.scalar("Loss", loss)

		build_model()


	def train(self, generator,restore):

		saver = tf.train.Saver()


		with tf.Session() as sess:
			try:

				run_id = np.random.randint(0,1e7)

				train_writer = tf.summary.FileWriter(logdir="logs/" + str(run_id), graph=sess.graph)

				if restore:
					saver.restore(sess, tf.train.latest_checkpoint('./saves'))
				else:
					sess.run(tf.global_variables_initializer())

				counter = 0

				while True:

						counter += 1
						merge = tf.summary.merge_all()

						x,y = generator.__next__()
						_, summary = sess.run([self.optimize,merge],feed_dict={})

						train_writer.add_summary(summary,counter)

			except KeyboardInterrupt:
				print("Interupted... saving model.")
			
			save_path = saver.save(sess, "./saves/model.ckpt")


	def test(self,inputs,labels,restore):

		with tf.Session() as sess:

			if restore:
				saver.restore(sess, tf.train.latest_checkpoint('./saves'))
			else:
				sess.run(tf.global_variables_initializer())


			output = sess.run([self.output_distribution], feed_dict={self.x_placeholder:inputs})






