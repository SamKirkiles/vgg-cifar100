import tensorflow as tf
from vgg import VGG
import matplotlib.pyplot as plt
from data_loader import Loader

def main():
	print("running main")

	#vgg = VGG()	
	#vgg.train(False)
	loader = Loader()

	with tf.Session() as sess:

		iterator = loader.get_dataset()
		element = iterator.get_next()

		x,y = sess.run(element)

		#print(x)
		#plt.imshow(x[0])
		#plt.show()



if __name__ == "__main__":
	main()