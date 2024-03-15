# S7 Assignment

As part of this assignment we need to build a model with
* less than 8000 parameters
* 99.4% test accuracy in atleast two epochs consistently in <= 15 epochs

To achieve this as part of this project, we have 2 files.

***model.py*** : This file contains all the neural network models used for 3 steps. The model used for Step-1 is Model_1, Step-2 is Model_2 and Step-3 is Model_3.

***S7.ipynb***: This is ipynb file in which we trained and evaluated 3 steps using 3 models. We had it in one notebook as we can reuse some portion of code for others.

This problem needs to be solved in 3 steps.

So we have divided 1st 4 Code Tasks into Step1, Next 3 Code Tasks into Step2 and Last 3 into Step3.

## Step-1

* Code 1: Setup
* Code 2: Basic Skeleton
* Code 3: Lighter Model
* Code 4: Batch Normalization

### Model 1 Architecture

Using above 4 tasks/points we were able to build a model with 8.172k parameters. 

| Input | Layer | Output | Kernel | RF |
| --- | --- | --- | --- | --- |
| 28x28x1 | Conv1 | 26x26x4 | 3x3 |  3 |
| 26x26x4 | Conv2 | 24x24x8 | 3x3 | 5 |
| 24x24x8 | Conv3 | 22x22x16 | 3x3 |  7 |
| 22x22x16 | MaxPool1 | 11x11x16 | 2x2 |  8 |
| 11x11x16 | Conv4 | 11x11x4 | 1x1 |  8 |
| 11x11x4 | Conv5 | 9x9x8 | 3x3 | 12 |
| 9x9x8 | Conv6 | 7x7x16 | 3x3 |  16 |
| 7x7x16 | Conv7 | 5x5x10 | 3x3 |  20 |
| 5x5x10 | Conv8 | 1x1x10 | 5x5 |  28 |


No of parameters: 8,172

## Target, Result and Analysis of Step-1

#### Target:
    Add Batch normalization to improve model efficiency.
#### Results:
    Parameters : 8.17k
    Best Train Accuracy : 99.73%
    Best Test Accuracy : 99.26%
#### Analysis:
    Before adding batch normalization we had best test accuracy as 98.9% and best training accuracy as 99.5%.
    Upon adding batch norm, we have started to see over-fitting now.
    Even if the model is pushed further, it won't be able to get to 99.4%

## Step-2

* Code 5: Regularization
* Code 6: Global Average Pooling
* Code 7: Increasing Capacity

### Model 2 Architecture

We had the problem with overfitting in Model1 and we were not able to achieve targetted accuracy of 99.4%. Here we use Regularization techniques like dropout to reduce gap between training and test accuracy.

Using above 3 tasks/points we were able to build a model with 8.084k parameters. 

| Input | Layer | Output | Kernel | RF |
| --- | --- | --- | --- | --- |
| 28x28x1 | Conv1 | 26x26x4 | 3x3 |  3 |
| 26x26x4 | Conv2 | 24x24x8 | 3x3 | 5 |
| 24x24x8 | Conv3 | 22x22x16 | 3x3 |  7 |
| 22x22x16 | MaxPool1 | 11x11x16 | 2x2 |  8 |
| 11x11x16 | Conv4 | 11x11x4 | 1x1 |  8 |
| 11x11x4 | Conv5 | 9x9x8 | 3x3 | 12 |
| 9x9x8 | Conv6 | 7x7x16 | 3x3 |  16 |
| 7x7x16 | Conv7 | 5x5x32 | 3x3 |  20 |
| 5x5x32 | Conv8 | 5x5x10 | 1x1 |  20 |
| 5x5x10 | GAP | 1x1x10 | 5x5 |  28 |

No of parameters: 8,084

Here we have used Dropout as 0.25 which was used after Conv3 and Conv6.

## Target, Result and Analysis of Step-2

#### Target:
    Add Dropout Layer
    Add GAP Layer to remove big 7x7 kernel
    Increase capacity of model. Add layers at the end.
#### Results:
    Parameters : 8.084k
    Best Train Accuracy : 98.93%
    Best Test Accuracy : 99.09%
#### Analysis:
    Add dropout has reduced the difference between Training and testing accuracy.
    Adding GAP helped in reducing no of parameters from 8.1k to 3.2k.
    As test accuracy was decreased with reduction in parameters, we added capacity at the end layers.
    Model was not overfitting.
    Even if the model is pushed further, it won't be able to get to 99.4%.

## Step-3

* Code 8: Correct Max Pooling Location
* Code 9: Image Augmentation
* Code 10: Playing naively with learning rates

### Model 3 Architecture

we were not able to achieve targetted accuracy of 99.4% with Model2. Here we need to adjust max pooling location as it should be done after 1x1 convolution in the transition block.

Using above 3 tasks/points we were able to build a model with 7.72k parameters. 

| Input | Layer | Output | Kernel | RF |
| --- | --- | --- | --- | --- |
| 28x28x1 | Conv1 | 26x26x8 | 3x3 |  3 |
| 26x26x8 | Conv2 | 24x24x16 | 3x3 | 5 |
| 24x24x16 | Conv3 | 24x24x8 | 1x1 | 5 |
| 24x24x8 | MaxPool1 | 12x12x8 | 2x2 |  6 |
| 12x12x8 | Conv4 | 10x10x16 | 3x3 |  10 |
| 10x10x16 | Conv5 | 8x8x16 | 3x3 | 14 |
| 8x8x16 | Conv6 | 6x6x16 | 3x3 |  18 |
| 6x6x16 | Conv7 | 6x6x16 | 1x1 |  18 |
| 6x6x16 | GAP | 1x1x16 | 6x6 |  28 |
| 1x1x16 | Conv8 | 1x1x10 | 1x1 |  28 |

No of parameters: 7720

We have used scheduler as ReduceLROnPlateau.

## Target, Result and Analysis of Step-3 ( Final Model)

#### Target:
    Adjust Max pool location
    Add Image augmentation like Rotation of image
    Change learning rate (  adaptive learning rate )
#### Results:
    Parameters : 7.72k
    Best Train Accuracy : 98.85%
    Best Test Accuracy : 99.40% ( 13th and 14th epoch)
#### Analysis:
    Adjusting max pooling in transition layer and adding dropout with value 0.05 on each layer made model not to overfit at all.
    We have added more layers at the end which improved accuracy.
    As per data, slight rotation of images is required ( we used 15 degrees) which will make training harder and test results better.
    As we have used ReducedLROnPlateau, we are decreasing LR based on the decrease in validation/test loss which led to achieve 99.4% consistently in last 2-3 epochs.
    
