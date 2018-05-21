import tensorflow as tf
from vgg import VGG
from data_loader import Loader
import matplotlib.pyplot as plt

def main():
	print("running main")

	vgg = VGG()

	# Load CIFAR-100 dataset
	loader = Loader()

	generator = loader.get_train_batch()
	
	vgg.train(generator,False)

	"""
	test_y_one_hot = np.eye(test_y.shape[0])[test_y]
	train_y_one_hot = np.eye(train_y.shape[0])[np.ravel(train_y)]
	"""

if __name__ == "__main__":
	main()