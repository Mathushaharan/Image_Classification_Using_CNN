# Image_Classification_Using_CNN

Project Description:
This project is an implementation of image classification using Convolutional Neural Networks (CNN) on the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The goal is to train a CNN model to accurately classify these images into their respective categories.

CNN model description:
The model consists of several convolutional layers, followed by max-pooling layers.
After the convolutional layers, there are dense layers for classification.

The model's architecture:
Three convolutional layers with increasing filter sizes (32, 64, and 128) and ReLU activation functions.
Max-pooling layers after each convolutional layer to downsample the feature maps.
Flattening layer to convert the 2D feature maps into a 1D vector.
Three dense layers with ReLU activation functions and dropout layers for regularization.
The final output layer with 10 units and softmax activation for multi-class classification.

Model Compilation:
The model is compiled with an optimizer, loss function, and evaluation metric.
The optimizer and loss are defined externally and should be set before this code.
It uses the 'sparse_categorical_crossentropy' loss function for multi-class classification.

Early Stopping:
Early stopping is implemented to prevent overfitting during training.
It monitors the validation loss and stops training if the loss stops decreasing (with a small threshold).
The patience parameter specifies how many epochs to wait before stopping.
The restore_best_weights parameter determines whether to restore the model weights to the best seen during training.
Early stopping callbacks are passed to the fit method.

Training:
The model is trained using the fit method.
It is trained for 1000 epochs but will stop early if the validation loss doesn't improve for 20 consecutive epochs.
The training data (X_train and Y_train) are used, and a validation split of 30% is specified.
A batch size of 30 is used for mini-batch training.
