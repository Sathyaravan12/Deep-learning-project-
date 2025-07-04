# Deep-learning-project

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: SATHYA MURTHY E

*INTERN ID*: CT08DK896

*DOMAIN*: DATA SCIENCE

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTHOSH

 Project Description – Task 2: Deep Learning Image Classification Using CNN
In today’s digital world, image classification plays a crucial role in various industries such as healthcare, autonomous vehicles, security, and social media. One of the most powerful techniques for image classification is Convolutional Neural Networks (CNNs), a class of deep learning models specifically designed to process pixel data. In this project, we leverage a CNN built with TensorFlow and Keras to classify images from the CIFAR-10 dataset into one of ten categories.

The CIFAR-10 dataset is a standard benchmark in the field of computer vision. It consists of 60,000 images categorized into 10 classes such as airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Each image is a small color picture with a resolution of 32x32 pixels and three color channels (RGB). The dataset is split into 50,000 training images and 10,000 testing images, making it an ideal choice for training and validating deep learning models in a limited computing environment.

Our approach begins with importing and normalizing the dataset. Normalization ensures that pixel values are scaled to a range between 0 and 1, improving model training performance. Following this, we build a CNN model using the Keras Sequential API. The architecture includes three convolutional layers with ReLU activation functions and max-pooling operations to reduce spatial dimensions while preserving important features. The output from these layers is flattened and passed through a fully connected dense layer, ending with an output layer consisting of 10 neurons (one for each class).

The model is compiled using the Adam optimizer and Sparse Categorical Crossentropy loss function. It is trained over 10 epochs, using the testing set for validation. During training, we monitor accuracy and loss for both the training and validation sets to evaluate the model’s learning progress and generalization capability.

After training, we evaluate the model on the test set and visualize its performance using two plots: Accuracy vs Epoch and Loss vs Epoch. These plots help us understand whether the model is overfitting or underfitting. In our case, the model achieved approximately 75% validation accuracy and 79% training accuracy, which is a strong result for a basic CNN without data augmentation or transfer learning.

This project demonstrates the effectiveness of CNNs for image classification tasks. It also highlights the simplicity and power of TensorFlow/Keras for building deep learning applications. While the model performs well, there is room for improvement through techniques such as data augmentation, dropout regularization, and experimenting with deeper architectures or pre-trained models like ResNet or VGG16. 

In conclusion, this task successfully showcases a complete deep learning pipeline for image classification, from data preprocessing to model evaluation and visualization. It serves as a foundational project for any aspiring data scientist or machine learning engineer interested in computer vision.
