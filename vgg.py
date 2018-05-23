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
				conv1 = tf.layers.conv2d(x, filters=64,kernel_size=(3,3),padding='SAME',
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
				bn_1 = tf.contrib.layers.batch_norm(conv1,activation_fn=tf.nn.relu,is_training=training)
				# Layer2 - 64 channels
				conv2 = tf.layers.conv2d(bn_1, filters=64,kernel_size=(3,3),padding='SAME',
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())

				bn_2 = tf.contrib.layers.batch_norm(conv2,activation_fn=tf.nn.relu,is_training=training)

				pool2 = tf.layers.max_pooling2d(bn_2, (2,2), (2,2), padding='SAME')

				#Layer 3 - 128 channels
				conv3 = tf.layers.conv2d(pool2, filters=128,kernel_size=(3,3),padding='SAME',
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())

				bn_3 = tf.contrib.layers.batch_norm(conv3,activation_fn=tf.nn.relu,is_training=training)

				# Layer 4 - 128 channels
				conv4 = tf.layers.conv2d(bn_3, filters=128,kernel_size=(3,3),padding='SAME',
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
				bn_4 = tf.contrib.layers.batch_norm(conv4,activation_fn=tf.nn.relu,is_training=training)
				pool4 = tf.layers.max_pooling2d(bn_4, (2,2), (2,2), padding='SAME')

				#Layer 5 - 256 channels
				conv5 = tf.layers.conv2d(pool4, filters=256,kernel_size=(3,3),padding='SAME',
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
				bn_5 = tf.contrib.layers.batch_norm(conv5,activation_fn=tf.nn.relu,is_training=training)
				# Layer 6 - 256 channels
				conv6 = tf.layers.conv2d(bn_5, filters=256,kernel_size=(3,3),padding='SAME',
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
				bn_6 = tf.contrib.layers.batch_norm(conv6,activation_fn=tf.nn.relu,is_training=training)
				# Layer 7 - 256 channels
				conv7 = tf.layers.conv2d(bn_6, filters=256,kernel_size=(3,3),padding='SAME',
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
				bn_7 = tf.contrib.layers.batch_norm(conv7,activation_fn=tf.nn.relu,is_training=training)
				pool7 = tf.layers.max_pooling2d(bn_7, (2,2), (2,2), padding='SAME')

				# Layer 8 - 512 channels
				conv8 = tf.layers.conv2d(pool7, filters=512,kernel_size=(3,3),padding='SAME',
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
				bn_8 = tf.contrib.layers.batch_norm(conv8,activation_fn=tf.nn.relu,is_training=training)
				# Layer 9 - 512 channels
				conv9 = tf.layers.conv2d(bn_8, filters=512,kernel_size=(3,3),padding='SAME',
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
				bn_9 = tf.contrib.layers.batch_norm(conv9,activation_fn=tf.nn.relu,is_training=training)
				# Layer 10 - 512 channels
				conv10 = tf.layers.conv2d(bn_9, filters=512,kernel_size=(3,3),padding='SAME',
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
				bn_10 = tf.contrib.layers.batch_norm(conv10,activation_fn=tf.nn.relu,is_training=training)
				pool10 = tf.layers.max_pooling2d(bn_10, (2,2), (2,2), padding='SAME')

				# Layer 11 - 512 channels
				conv11 = tf.layers.conv2d(pool10, filters=512,kernel_size=(3,3),padding='SAME',
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
				bn_11 = tf.contrib.layers.batch_norm(conv11,activation_fn=tf.nn.relu,is_training=training)
				# Layer 12 - 512 channels
				conv12 = tf.layers.conv2d(bn_11, filters=512,kernel_size=(3,3),padding='SAME',
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
				bn_12 = tf.contrib.layers.batch_norm(conv12,activation_fn=tf.nn.relu,is_training=training)
				# Layer 13 - 512 channels
				conv13 = tf.layers.conv2d(bn_12, filters=512,kernel_size=(3,3),padding='SAME',
					use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
				bn_13 = tf.contrib.layers.batch_norm(conv13,activation_fn=tf.nn.relu,is_training=training)

				pool13 = tf.layers.max_pooling2d(bn_13, (2,2), (2,2), padding='SAME')


				flattened = tf.contrib.layers.flatten(pool13)

				dense14 = tf.layers.dense(inputs=flattened, units=4096,kernel_initializer=tf.contrib.layers.xavier_initializer())
				bn_14 = tf.contrib.layers.batch_norm(dense14,activation_fn=tf.nn.relu,is_training=training)
				dense15 = tf.layers.dense(inputs=bn_14, units=4096,kernel_initializer=tf.contrib.layers.xavier_initializer())
				bn_15 = tf.contrib.layers.batch_norm(dense15,activation_fn=tf.nn.relu,is_training=training)
				dense16 = tf.layers.dense(inputs=bn_15, units=100,activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer())


				# Predict
				scaled_logits = -tf.log(dense16)
				prediction = tf.argmax(tf.nn.softmax(scaled_logits),axis=1)

				equality = tf.equal(prediction, y)
				accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

				# Train
				softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=dense16,name="softmax")

				loss = tf.reduce_mean(softmax)

				optimize = tf.train.AdamOptimizer().minimize(loss)

			self.loss = loss
			self.x_placeholder = x
			self.y_placeholder = y
			self.training = training
			self.accuracy = accuracy
			self.optimize = optimize

			tf.summary.scalar("Loss", loss)

		build_model()


	def train(self,restore=False):

		saver = tf.train.Saver()


		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)) as sess:
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
				
				merge = tf.summary.merge_all()

				while True:

						counter += 1

						_, summary = sess.run([self.optimize,merge],feed_dict={})

						train_writer.add_summary(summary,counter)

						if counter%1000 == 0:

							# Check validation accuracy
							acc = sess.run([self.accuracy],feed_dict={self.x_placeholder:val_x,self.y_placeholder:val_y,self.training:False})

							accuracy_summary = tf.Summary(value=[tf.Summary.Value(tag='Test Accuracy',simple_value=acc)])
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






