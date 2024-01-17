# Project Name
> Build a convolution neural network, CNN, model to which detect melanoma, which is a type of cancer that can be deadly if not detected early. The problem is designed as a multiclass classification to predict mostly likely label among 9 exists in the dataset. 


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
- Provide general information about your project here.
- What is the background of your project?
    To build a CNN based model which can accurately detect melanoma. It accounts for 75% of skin cancer deaths.
- What is the business probem that your project is trying to solve?
    A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.
- What is the dataset that is being used?
    The dataset consists of 2357 images of malignant and benign oncological diseases (categoried into 9 diseases).

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Conclusions
- Optimized CNN architecture has dominated impact on model performaned. Train model with raw images provided in the dataset achieved accuracy of 60.7%.
- Training with weighted cross-entropy loss have marginal impact on the model perforamance. The accuracy drops to 59.85%.
- Training the model with augmented images did not improve the model performance. Accuracy of the model was only 59.85%.
    - Applied rotation and zoom augmentation for all the images to address class imbalance issue.

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- PyTorch - version 2.1
- numpy - version 1.23.5
- pandas - version 1.5.3
- matplotlib - version 3.7
- seaborn - version 0.12.2
- sklearn - version 1.2.1
- augmentor - version 0.6.0

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Acknowledgements


## Contact
Created by [@nagarajuoruganti] - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->