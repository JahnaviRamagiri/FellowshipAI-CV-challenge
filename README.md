# Computer Vision Challenge: Flower Classification

This README provides an overview of the solution for the computer vision challenge, which involves training a pre-trained ResNet 50 model on the Flowers dataset for flower classification.

## Problem Statement

The goal of this challenge is to develop a model that accurately classifies different flower species based on input images. I am provided with the Flowers dataset, which contains a collection of images belonging to various flower categories.

## Approach

To tackle the challenge, I employed the following approach:

1. **Data Preparation:** I split the Flowers dataset into training, validation, and test sets. This ensures that I have separate datasets for training the model, evaluating its performance during development, and testing the final trained model's accuracy on unseen data.

2. **Pre-trained Model:** I utilized the ResNet 50 architecture as a pre-trained model. The pre-trained model has been trained on large-scale image datasets and can extract meaningful features from images.

3. **Fine-tuning:** I performed fine-tuning by retraining the last few layers of the pre-trained ResNet 50 model on the Flowers dataset. This process allows the model to adapt its learned representations to the specific task of flower classification. Further, a Sequential Model of 2 layers was added at the end of the pre-trained model architecture.

4. **Training and Optimization:** I trained the model using the training set and monitored its performance on the validation set. Techniques such as data augmentation, feature scaling, and regularization are employed to improve generalization and prevent overfitting.

5. **Model Evaluation:** After training, the model's performance on the test set is evaluated. GRADCAM outputs show the misclassified predictions and their gradients.

## Dataset
The Flowers-102 dataset consists of 102 different categories of flowers, with each category containing varying numbers of images. The dataset is a collection of flower images from various species, such as roses, sunflowers, daisies, tulips, and more.

Image Count: The dataset contains a total of 8,189 images, with each image typically representing a single flower.

Image Variety: The dataset covers a wide range of flower species, showcasing the diversity of floral forms, colors, and textures. Each flower category has a different number of images, resulting in imbalanced class distribution.

Image Resolution: The images in the Flowers-102 dataset have varying resolutions and aspect ratios. Some images may be higher or lower in resolution compared to others.

Labeling: Each image in the dataset is assigned a unique label indicating the flower category it belongs to. These labels range from 1 to 102, representing the different flower classes.

## Data Visualization

![image](https://github.com/JahnaviRamagiri/FellowshipAI-CV-challenge/assets/61361874/324adf81-cbf3-4f58-aff4-5fb0b6e30381)


![image](https://github.com/JahnaviRamagiri/FellowshipAI-CV-challenge/assets/61361874/d8406420-ed93-4209-b844-88ebb5ea666e)

![image](https://github.com/JahnaviRamagiri/FellowshipAI-CV-challenge/assets/61361874/31dfab04-899b-4aea-9d8b-997071ce9a41)


## Observations and Potential Challenges
1.  Class Imbalance: The Flower-102 dataset is known to have imbalanced class distributions, meaning that some flower categories may have a significantly higher number of images compared to others. This imbalance can affect the model's training and performance, as it may be biased towards the majority classes.

2. Varied Flower Species: The dataset encompasses a diverse range of flower species, including roses, sunflowers, daisies, tulips, lilies, orchids, and many more. Each flower category represents a distinct species or variety, allowing for the exploration and classification of various floral forms.

3. Color and Shape Variations: The dataset captures the diversity of flower colors and shapes. Different flower categories exhibit variations in color palettes, ranging from vibrant reds and yellows to soft pinks and whites. Additionally, flower shapes can vary significantly, with some categories displaying round petals, while others may have elongated or irregular shapes.

4. Similarity and Confusion: Within the 102 classes, there might be certain flower categories that share visual similarities, making it challenging to distinguish between them. For instance, different varieties of roses or tulips might have similar appearances, requiring more precise analysis and classification techniques.

5. Occlusion and Background Variations: The images in the dataset may contain occlusions, such as leaves or other objects partially covering the flowers. Additionally, the background settings can vary, ranging from natural outdoor scenes to controlled indoor environments. These variations add complexity to the task of flower classification and require models to learn robust features.

6. Intra-class Variability: Each flower category may have inherent variability in terms of size, petal arrangement, and bloom stage. Some categories might include images of flowers at different growth stages or with varying degrees of bloom, which adds another layer of complexity to the classification task.



## Data Augmentation

For data augmentation, I utilized the Albumentations library, which offers a wide range of image augmentation techniques. Albumentations provide efficient and flexible transformations such as random rotations, flips, zooms, and color adjustments. By applying these augmentations, the diversity of the dataset is improved, helping the model generalize better and reducing overfitting.

```python
tr_trans = [
              alb.Resize(height=256, width=256),
              alb.ShiftScaleRotate(shift_limit=0.4, scale_limit=0.5, rotate_limit=45, p=0.5),
              alb.GaussianBlur(blur_limit=(3, 7), p=0.5),
              alb.HorizontalFlip(p= 0.75),
              alb.GaussNoise(var_limit=(0.01, 0.1), p=0.5),
              alb.VerticalFlip(p=0.5),
              alb.RandomBrightnessContrast (brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
              alb.CenterCrop(224,224, always_apply= True),
              alb.CoarseDropout(max_holes=1, max_height=70, max_width=70, p=0.5),
              alb.Normalize(
                  mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]
              ),
              ToTensor()
              ]

trans = Flowers102_AlbumTrans(tr_trans)
data = FLOWER102DataLoader(trans, batch_size=32)
train_loader, test_loader = data.get_loaders()
display(train_loader, 32)
```
```tr_trans``` shows image transformations using the Albumentations library, including resizing, rotation, blurring, flipping, noise addition, brightness/contrast adjustments, cropping, and normalization. These transformations are then applied to the Flower-102 dataset using a custom transformation class. The dataset is loaded into data loaders with a batch size of 32, and the train loader is displayed.

## Class Imbalance and Overfitting

During the classification task, it is crucial to consider class imbalance and overfitting:

- **Class Imbalance:** Class imbalance refers to a situation where the number of examples in each class is not evenly distributed. It can lead to biased model predictions, with higher accuracy on the majority class and poor performance on minority classes. Techniques such as oversampling, undersampling, or using class weights can help address the class imbalance.

- **Overfitting:** Overfitting occurs when a model learns to perform well on the training data but fails to generalize to unseen data. Signs of overfitting include high accuracy on the training set but poor performance on the validation or test sets. Regularization techniques, such as dropout and weight decay, can mitigate overfitting.

## RESNET50 : RESNET 50 architecture pretrained on IMAGENET is used for this experiment. Last 2 layers are Trainable and the other layers are freezed.
## Experiment 1
### Model
![image](https://github.com/JahnaviRamagiri/FellowshipAI-CV-challenge/assets/61361874/d83b677c-b338-4b6e-9323-072afedbebce)


### Misclassified Images:
![image](https://github.com/JahnaviRamagiri/FellowshipAI-CV-challenge/assets/61361874/db1a6cac-00f1-4417-9e6a-8408c693435f)


## Experiment 2

![image](https://github.com/JahnaviRamagiri/FellowshipAI-CV-challenge/assets/61361874/275e224c-8ea8-4ecd-9fd6-1b2b04b20edb)


### Misclassified Images
![image](https://github.com/JahnaviRamagiri/FellowshipAI-CV-challenge/assets/61361874/12b97362-44f8-4518-8ef6-189989b199c2)

# Accuracy
Experiment 1

![image](https://github.com/JahnaviRamagiri/FellowshipAI-CV-challenge/assets/61361874/66e826cc-854e-4298-bbff-22af772dfb13)

Experiment 2

![image](https://github.com/JahnaviRamagiri/FellowshipAI-CV-challenge/assets/61361874/69525017-d861-4d53-ab4e-608d5ce04f60)

Experiment 2 has a significant decrease in Overfitting when compared to Experiment 1. Accuracy has also increased by 3% to 98.04 from an initial 95%


# GRADCAM
![image](https://github.com/JahnaviRamagiri/FellowshipAI-CV-challenge/assets/61361874/da8b1a92-12f4-4498-b958-ea4f0f1c7644)
![image](https://github.com/JahnaviRamagiri/FellowshipAI-CV-challenge/assets/61361874/473ed141-dfdc-4241-a7fb-eefdcf6d7797)


## State-of-the-Art (SOTA)
The SOTA for the Flowers102 dataset is with Compact Convolutional Transformer at 99.76%. The other SOTA models are displayed below:
![image](https://github.com/JahnaviRamagiri/FellowshipAI-CV-challenge/assets/61361874/2f652d5f-cd28-42c7-9060-33ecf0c3a5b8)


Given the pre-trained architecture we have considered in this challenge, the SOTA for a RESNET50 model is 97.9%. Our model has achieved 98.04 % with 24.3 M Trainable Parameters.
![image](https://github.com/JahnaviRamagiri/FellowshipAI-CV-challenge/assets/61361874/f9a73897-df97-4a0b-9ad2-535f54afc4f7)



## Bugs in the Code

To ensure the absence of bugs and maintain the reliability of the code, the following steps are implemented:

- Implement comprehensive unit tests to verify the correctness of individual components and functions in the code.
- Split the dataset into training, validation, and test sets and monitor the model's performance on the validation set during training to ensure it's not overfitting.
- Compare the model's performance with existing benchmarks or previous results to ensure it falls within a reasonable range.


## Future Scope

While our model achieved satisfactory results, there are several avenues for future improvement and exploration:

- **Dataset Size:** The Flowers dataset is relatively small, which can limit the model's ability to generalize well. Expanding the dataset with additional images or considering larger-scale datasets may enhance performance.
- **Hyperparameter Tuning:** Further exploration of hyperparameter tuning, such as learning rate, batch size, and regularization techniques, could potentially improve the model's accuracy.
- **Architecture Modifications:** Experimenting with different network architectures or variations of the ResNet model, such as ResNet with more layers (e.g., ResNet 101 or ResNet 152), may capture more intricate features and boost performance.
- **Ensemble Learning:** Combining predictions from multiple models or using ensemble learning techniques, such as bagging or boosting, could potentially improve the model's robustness and accuracy.



