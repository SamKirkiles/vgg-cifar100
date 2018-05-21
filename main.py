import tensorflow as tf
from vgg import VGG
from data_loader import Loader
import matplotlib.pyplot as plt

def main():
	print("running main")

	model = VGG()

	# Load CIFAR-100 dataset
	loader = Loader()

	generator = loader.get_batch()
	x,y = generator.__next__()
	plt.imshow(x[0])
	plt.show()

	"""
	test_y_one_hot = np.eye(test_y.shape[0])[test_y]
	train_y_one_hot = np.eye(train_y.shape[0])[np.ravel(train_y)]
	"""

if __name__ == "__main__":
	main()