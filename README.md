# Ad-Clicks
Feature Engineering and Simple Neural Network Modeling

The dataset used in this project can be found on https://www.kaggle.com/datasets/gabrielsantello/advertisement-click-on-ad

The goal of this project was to use features (and create new ones) from the dataset to predict whether a particular person had clicked on an online ad or not.

This project includes:

*Feature Engineering

*Exploratory Data Analysis

*Data Visualization

*Simple Neural Network Classification

This project is focused on building a neural network to predict whether or not a user clicked on an advertisement, using a dataset of user features and a binary 'Clicked on Ad' column. The dataset is obtained from Kaggle and contains the following features: daily time spent on site, age, area income, daily internet usage, ad topic line, city, gender, country, timestamp, and clicked on ad.

Feature engineering is carried out to generate the new features. The 'Country' column is used to create a new categorical feature, 'Continent', and the 'Timestamp' column is used to create two new features, Month and Hour. The 'Ad Topic Line' column is dropped, and the City, Country, and Timestamp columns are also dropped, leaving the cleaned dataset with new features.

Exploratory data analysis is then carried out to understand the relationships between the different features in the dataset. The target class, 'Clicked on Ad', is found to be evenly distributed in the dataset, which means that there is no class bias. Histograms are plotted to visualize the distribution of different features in the dataset, which shows that the numerical features are well separated between the classes. The categorical features are also compared against the numerical features to identify their separation boundaries.

A simple neural network is built to predict if a user clicked on an advertisement. The dataset is split into training and testing sets, and a sequential neural network is built using Keras. The model consists of an input layer, a hidden layer with 4 nodes, and an output layer. The model is trained on the training set and tested on the testing set, and the accuracy of the model is found to be 95.4%. The model is then used to make predictions on new data.

In the next part of the project, the focus is on finding the optimal number of neurons to use for the dense layer of the neural network. To accomplish this, a function called "optimize_neurons" is created, which trains and predicts through a range of neurons and selects the number of neurons that returned the highest accuracy. The function requires the training and test data, the number of neurons to test, and the activation function to use.

The function trains the model using the specified number of neurons and activation function and computes the accuracy on the test data. It does this for a range of neurons and selects the number of neurons that gave the highest accuracy. The function prints the iteration number as it progresses and returns the optimal number of neurons and the best accuracy achieved.

The function is run three times with different activation functions ('relu', 'softmax', and 'selu') and a fixed number of neurons (40). The results show that the best combination of neurons and activation function is 3 neurons with the 'relu' activation function. This combination is used for the final model.

The final model architecture consists of a dense layer with 3 neurons and a 'relu' activation function, followed by a final layer for binary classification with a sigmoid activation function. The model is compiled with binary crossentropy loss and the Adam optimizer. The model is trained for 500 epochs with early stopping to prevent overfitting. The test data is used for validation during training.

The performance of the final model is evaluated using the classification report and the confusion matrix on the test data. The results show an accuracy of 0.97, precision of 0.95 and 0.98, recall of 0.95 and 0.98, and f1-score of 0.97 for both classes. The confusion matrix shows that the model correctly classified 124 out of 130 samples of the negative class and 118 out of 120 samples of the positive class.

To generalize the performance of the model across multiple train/test splits, a function called "Avg_Model_Accuracy" is created. The function uses the data, the target variable, the name of the model, the scaler to use, the number of splits to perform, the test size, and a flag to plot a kernel density estimate of the accuracy distribution. The function splits the data into train and test sets for each iteration, scales the data, builds the model, and computes the accuracy on the test data. The function returns the average accuracy across all splits and the standard deviation.

After exploring the data and generating some new features, I was able to build a model that can predict if a person clicked on an ad with greater than 95% accuracy! We can use this model on new data points to determine if that person would click the proposed ad or not. We could use this information to display the ad to the users most likely to click it and then collect data on whether or not they did click the ad to see how well our model performs. If the model performs well, we can continue to use it to build a database of target customers (people that click on the ad) and ensure that these people are being targeted by future marketing campaigns. If the model does not perform well, we should try some other models on the new data and repeat the process until we find a model that meets our standards.
