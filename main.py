import tensorflow as tf
from vgg import VGG
import matplotlib.pyplot as plt
from data_loader import Loader

def main():

	vgg = VGG()	
	# SAVES MUST BE DOWNLOADED FOR RESTORE TO WORK
	vgg.train(restore=True)
	#vgg.test(restore=True)

if __name__ == "__main__":
	main()