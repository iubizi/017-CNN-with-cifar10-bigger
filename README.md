# 017-CNN-with-cifar10-bigger

017 CNN with cifar10 bigger

## training situation

![vgg16](https://github.com/iubizi/017-CNN-with-cifar10-bigger/blob/main/visualization.jpg)

## overfitting

Looking at the loss, we can see an obvious overfitting problem. Although the accuracy of the subsequent models has improved, the loss has become larger and larger, so the training should be stopped earlier. (The earlystopping setting is too large, in order to demonstrate overfitting)
