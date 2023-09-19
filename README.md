# BrainTumorDetection
INTRODUCTION:
In the realm of medical diagnostics, the accurate and timely detection of brain tumors plays a pivotal role in patient care and treatment outcomes. Traditional methods rely heavily on manual analysis of medical images, which can be time-consuming, prone to human error, and limited by the expertise of the interpreting physician. However, the landscape is rapidly evolving with the advent of cutting-edge technologies such as machine learning. 
The Brain MRI Images for Brain Tumor Detection dataset, which was available on Kaggle contained MRI images of the brain with and without tumors. The project comprises of implementing a machine learning algorithm and using transfer learning to generate two model fits and then comparing their accuracy.

TRANSFER LEARNING:
The choice for our transfer learning model was Inception V3 which is 48 layers deep. A pretrained version of this model is available on ImageNet database. It can classify objects into 1000 different classes. By leveraging the power of deep learning and convolutional neural networks (CNNs), Inception v3 offers a promising approach to accurately detect and classify brain tumors from medical imaging data. A detailed image representing the architecture of the Inception V3 model is given below.

PREPROCESSING IN INCEPTION V3:
The pretrained model Inception V3 requires that all the training images are received in a specific format. Here we are converting all the images into a 299 x 299 sized matrix of pixel values. The following chunk of code also contains data augmentation which is basically a technique to expand your dataset. Here we are applying multiple shifts and flips thus generating many varying copies of every image in our training set.
input_size = (299, 299)  # Input size for InceptionV3 model
batch_size = 32  # Number of images to process in each batch
data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

![image](https://github.com/subatakhan28/BrainTumorDetection/assets/145452943/7a43faea-4304-49f1-9d64-20269edb3801)



