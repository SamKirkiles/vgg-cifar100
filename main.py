import tensorflow as tf
from vgg import VGG
from data_loader import Loader
import matplotlib.pyplot as plt

def main():
	print("running main")

	vgg = VGG()	
	vgg.train(generator,False)


if __name__ == "__main__":
	main()