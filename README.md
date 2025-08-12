# Signature Validation - Model
This repository presents the implementation of a Siamese Convolutional Neural Network (SCNN) model for handwritten signature verification, a critical task in biometrics and document security. This project focuses on "offline" verification, analyzing signatures from static images to overcome the limitations of traditional methods.

#### Preparing the dataset
1. The model is trained on CEDAR dataset which can be downloaded from [here](http://www.cedar.buffalo.edu/NIJ/data/signatures.rar).
2. Extract the files and then you will get the following file structure :
```
|-- dataset/cedar1
|	|-- full_org
|	|	|-- original_1_1.jpg
|	|	|-- original_1_2.jpg
|	|	|-- ...(all original 24 signs of 24 signers i.e. 24x24 = 576 images)
|	|-- full_forg
|	|	|-- forgeries_1_1.jpg
|	|	|-- forgeries_1_2.jpg
|	|	|-- ...(all forged 24 signs of 24 signers i.e. 24x24 = 576 images)
```
3. To use the model, you need a CSV file named signature_pairs.csv that organizes the image pairs for training. This file must have a specific structure with three columns: img1, img2, and label.

    * **img1**: The full path to the first signature image.

    * **img2**: The full path to the second signature image.

    * **label**: A numerical value that indicates whether the signatures in img1 and img2 are from the same author (similar) or not.

        * 1: The signatures are genuine (from the same author).
        * 0: The signatures are forged (from different authors).

    
    Example: 
    ```
    img1,img2,label
    dataset/cedar1/full_org/original_10_1.png,dataset/cedar1/full_org/original_10_10.png,1
    dataset/cedar1/full_org/original_10_1.png,dataset/cedar1/full_org/original_10_11.png,1
    ...
    dataset/cedar1/full_org/original_10_1.png,dataset/cedar1/full_forg/forgeries_10_1.png,0
    ...
    ```

#### Training

All the necessary configurations and training logic are defined within the train.py file. To begin the training process, simply execute this script from your terminal:
```
python train.py
```
**Note:** We highly recommend reviewing the parameters and configurations inside train.py (such as learning rate, batch size, and dataset paths) before running the script to ensure it meets your specific needs.

Leveraging Colab and the Keras/TensorFlow ecosystem, the model was trained and reached a **99.66%** training accuracy. It is anticipated that higher performance can be attained by increasing the dataset size through data augmentation.

#### Loss Function
The model was trained using contrastive loss and the RMSprop optimizer.

```
def contrastive_loss(y_true, y_pred):
    margin = 1.0
    return K.mean(y_true * K.square(y_pred) +
                (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
```
