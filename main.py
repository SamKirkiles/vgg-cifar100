import tensorflow as tf
from vgg import VGG
import matplotlib.pyplot as plt

def main():
	print("running main")
	model = VGG()

	# Load CIFAR-100 dataset
	(train_x,train_y), (test_x,test_y) = tf.keras.datasets.cifar100.load_data(label_mode='fine')


if __name__ == "__main__":
	main()