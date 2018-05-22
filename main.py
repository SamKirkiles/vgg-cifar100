import tensorflow as tf
from vgg import VGG
import matplotlib.pyplot as plt

def main():
	print("running main")

	vgg = VGG()	
	vgg.train(False)


if __name__ == "__main__":
	main()