import tensorflow as tf
from vgg import VGG
import matplotlib.pyplot as plt
from data_loader import Loader

def main():
	print("running main")

	vgg = VGG()	
	vgg.train(restore=False)



if __name__ == "__main__":
	main()