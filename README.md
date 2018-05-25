# VGG-16 on CIFAR-100

An VGG net (with batchnorm and dropout) trained on CIFAR-100. You can easily modify this code to train on CIFAR-10 by changing a line in the data loader class. Achieves around 64% accuracy without data augmentation. Record on this dataset is 75%. I plan to add data agumentation to get performance up to state of the art. 

Here's the architecture:

![Architecture](https://i.imgur.com/ibbfyos.png)

![Loss](https://i.imgur.com/8KuU0SG.png)
![Validation Accuracy](https://i.imgur.com/25lEbPK.png)
![Train Accuracy](https://i.imgur.com/SNSmvaO.png)

**Useful Links**
https://www.cs.toronto.edu/~kriz/cifar.html  
