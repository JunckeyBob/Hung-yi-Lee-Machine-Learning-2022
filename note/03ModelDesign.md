# Model Design
 - A complicate model gains small error on training dataset
 - A complicate model may gain large error on test dataset

## Source of Error
Function learned from machine learning is differet from the inherent function of dataset. The difference can be measured by error on dataset. **Bias** and **Variance** result in error. We need to make trade-off on bias and variance to minimize error.

### Bias
Different from **Model Bias**, bias measures the expectation of distance between ML function and the inherent function. 

Model is a function set, a simple model contains less function. So for a small function set, even the best funciton in that set may different a lot from the inherent function. 

### Variance
We can get different funcitons by training the same model on different datasets. The variance measures the spread between functions on different datasets. 

A complicate model is sensitive to dataset. So the variance of a complicate model is large. 

## Underfitting and Overfitting
 - Underfitting: Large error on both learing dataset and testing dataset
 - Overfitting: Small error on learning dataset, large error on testing dataset

### Smooth
Smoothness measures how sensitive to input the model is. 

Assuming the input changes, the less smooth the function, the greater the change in the output.

The closer the function parameters are to 0, the smoother the function is. 

**Why do we like a smooth function?**

A moderately smooth function can mitigate the impact of noise contained in the function input on the function output.

If the input contains some noise/interference, the output of the smoothing function is less affected by the noise contained in the input.

**Why don't we like overly smooth functions?**

If the function is too smooth, the features of the data cannot be effectively extracted. This is not the function we want.

Suppose there is an extremely smooth function, that is, the output of the function is not affected by the input, which is of course not a good function.

### Regularization
A method to offer smooth function. The purpose of involving $\lambda \sum_i (w_i)^2$ is to keep the values ​​of the function parameters as close to 0 as possible to make the function smoother.
$$
L_{new} = L_{old} + \lambda \sum_i (w_i)^2
$$

Regularization may increase Bias, so the parameters of regularization need to be adjusted.

## Cross Validation
In machine learning, it is often impossible to use all the data for model training, otherwise there will be no data set with which to evaluate the model.

### The Validation Set Approch
Divide the data set into two parts: training set (Training Set) and test set (Test Set).

The disadvantage of this method is that it relies on the method of dividing the training set and the test set, and only uses part of the data for model training.

### Leave One Out Cross Validation (LOOCV)
Assume that there are N data in the data set, take 1 of the data as the test set, and use the remaining N-1 data as the training set. Repeat this N times to get N models and N error values. Finally, use these N The average of the error values ​​evaluates the model.

This method is not affected by the method of dividing the training set and the test set, because each data has been tested separately; at the same time, this method uses N-1 data to train the model, and almost all the data is used to ensure the accuracy of the model. Bias are smaller.

The disadvantage of this method is that the amount of calculation is too large, which is N-1 times as time-consuming as The Validation Set Approach.

### K-fold Cross Validation
This method is a compromise of LOOCV, which divides the data set into K parts randomly.

The selection of K is a trade-off of Bias and Variance. Generally choose K=5 or 10.

The larger the K, the larger the amount of data in the training set during each training, and the smaller the Bias; but the greater the correlation between the training sets in each training (consider the most extreme case K=N, that is, LOOCV , the data used for each training is almost the same), this large correlation will cause the final error to have a greater Variance.