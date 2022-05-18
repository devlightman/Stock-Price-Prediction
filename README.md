# Stockman: Using Time Series Analysis to Predict the Future Price of Stock
Stock Market Price Prediction using Machine Learning and Neural Network that analyzes stock data for a specific company of choice for the past 60 days from Yahoo Finance and makes an estimated price prediction for the next day market price without having to import a dataset.

This program uses an artificial recurrent neural network, more specifically, the LSTM architecture to predict the closing stock price of a corporation using the past 60 days stock price. The program uses python as a programming language and was run on PyCharm Professional and Google Colaboratory as the IDE, but any other standard python IDE can be used to execute the code. 

For the libraries, math, numpy, pandas, pandas_datareader, MinMaxScaler from sklearn, Sequential from keras.models , Dense and LSTM from keras.layers were imported. The program imported matplotlib.pyplot as plt and use ‘fivethirtyeight’ as the plt style. The program uses yahoo finance to import the stock information and the start and end date, as well as the company ticker can be modified to the user’s discretion.  To make sure the program is functioning properly, the ticker ‘FB’ was used to import historical data for Meta Platforms, Inc. stock from January 1st, 2017, till November 30th, 2021, and run. Figure 1 is attached below.

<img width="368" alt="image" src="https://user-images.githubusercontent.com/20808296/168998127-1202f976-7846-4743-aeb3-fe179ed19d22.png">
Figure 1: Historical data for Meta Platforms, Inc. stock from January 1st, 2017, till November 30th, 2021

The program had successfully managed to import all the recorded data available for the selected timeline for Meta Platforms, Inc. The program can also visualize the closing price history for the selected ticker to give the data a visual representation.

<img width="468" alt="image" src="https://user-images.githubusercontent.com/20808296/168998507-c3688e52-62a3-4f47-8c86-7f05bc64530b.png">
Figure 2: Closing price of Meta Platforms, Inc. stock from January 1st, January 1st, 2017, till November 30th, 2021

Afterwards, the program proceeds to create a new data frame with only the ‘Close’ column and converts the data frame to a numpy array. To train the LSTM model on, the program gets the number of rows to train the model on by creating a variable called training_data_len and trains the data on 80% of the available data. Since it is always advantageous to apply preprocessing, transformations, scaling or normalization to the input data before it is presented to the neural network, the program uses MinMaxScaler to scale the data. The variable scaled_data holds the scaled dataset that is valued between 0 and 1. 

To create the training dataset, a scaled training dataset was created. The variable train_data contains all the values from index 0 to training_data_len and gets back all the columns. Next, the data was split into x_train and y_train datasets. The x_train is the independent training variables and y_train is the dependent or target variable. For the first pass through x_train will contain 60 values and those values will be indexed from position 0 to 59 and y_train dataset for the first pass through will contain the 61st value, which will be at position 60. In the next step, the x_train and y_train datasets were converted into numpy arrays. Because LSTM network expects the input to be three dimensional in the form of samples, number of time steps and number of features, and until this point, the x_train data set has been two dimensional, the x_train dataset was reshaped. Now that the input was converted to fit the LSTM architecture, the program proceeds to build the LSTM model and compile it. ‘Adam’ optimizer was used to improve upon the loss function. To train the model, batch size was set to 32 and epochs was set to 25. After training the model, a testing dataset was created and was converted to a numpy array and reshaped. Then the program proceeds to plot the data and the outcome in figure 3 shows the model visualize the trend. 

<img width="468" alt="image" src="https://user-images.githubusercontent.com/20808296/168998843-3f563f4e-f554-494d-b242-cff40e29727c.png">
Figure 3: Training, Actual Value and Prediction values for Meta Platforms, Inc. stock

The blue line shows the data the model was trained on, the orange line shows the actual closing price for the rest of the days and the yellow is the program’s predictions. For this project, the model was trained on 80% of the total imported training data, so the orange and yellow lines represent 20% of the graph. Since the actual price and the predicted price based on the trained data is very close, it can be considered that the model is showing decent results. The model also shows a table where it displays the actual price and the predicted price of Meta Platforms, Inc., which is shown on figure 4.

<img width="145" alt="image" src="https://user-images.githubusercontent.com/20808296/168998987-facd826c-6abd-4bbc-94d9-dd06c848e1bc.png">
Figure 4: The actual price and the predicted price of Meta Platforms, Inc.

<img width="161" alt="image" src="https://user-images.githubusercontent.com/20808296/168999042-71b87e00-465b-4ec9-b407-37fbc4ff0612.png">
Figure 5: The predicted price and the actual price of Meta Platforms, Inc. on December 27th, 2021

Upon taking a closer look at the predicted price and the actual price, it can be observed that the prediction was in very close range of the actual closing price of the stock.
 
The last part of the program obtains a quote on one day into the future of the chosen stock by analyzing the last 60 days closing price. To verify the actual closing price of the day, the actual closing price was obtained and compared to the predicted price. 





## Result

To check the program, closing stock price of Amazon.com Inc. (AMZN), NVIDIA Corporation (NVDA) and Apple Inc. (AAPL) were tested. The results are attached below.

<img width="160" alt="image" src="https://user-images.githubusercontent.com/20808296/168999357-88488343-2ab3-49ce-8c5e-327fc63e84cd.png">
Figure 6: The predicted and the actual price of AMZN on 2021-12-27. If we look on top, [[3398.2058]] is the predicted price from the model whereas 3393.389893 was the actual price on that day.

<img width="159" alt="image" src="https://user-images.githubusercontent.com/20808296/168999406-29a98d59-2c40-4610-abc6-f4969f9c83ef.png">
Figure 7: The predicted and the actual price of NVDA on 2021-12-27. If we look on top, [[288.41553]] is the predicted price from the model whereas 309.450012 was the actual price on that day.

<img width="156" alt="image" src="https://user-images.githubusercontent.com/20808296/168999441-6f5c526f-417e-4a79-91a4-fd5f2b790365.png">
Figure 8: The predicted and the actual price of AAPL on 2021-12-27. If we look on top, [[173.47493]] is the predicted price from the model whereas 180.330002 was the actual price on that day.

Although they do not predict the exact price into the future, considering this is the prediction from the neural net and by analyzing the data of a selected period, and the result is very close to the actual price, the outcome can be considered satisfactory.

The program was also tested to predict the closing stock price of Amazon.com Inc. (AMZN), NVIDIA Corporation (NVDA) and Apple Inc. (AAPL) based on the training data of Meta Platforms, Inc. The results are attached below. 

<img width="211" alt="image" src="https://user-images.githubusercontent.com/20808296/168999566-120a1768-8cb3-491f-8006-f1faef1c310f.png">
Figure 9: The predicted and the actual price of AMZN on 2021-12-27. If we look on top, [[1530.1288]] is the predicted price from the model whereas 3393.389893 was the actual price on that day.

<img width="217" alt="image" src="https://user-images.githubusercontent.com/20808296/168999608-972788eb-5fc3-4556-b358-60a786bbed0b.png">
Figure 10: The predicted and the actual price of AMZN on 2021-12-27. If we look on top, [[290.72168]] is the predicted price from the model whereas 3393.389893 was the actual price on that day.

<img width="221" alt="image" src="https://user-images.githubusercontent.com/20808296/168999660-9004a8c0-b29d-414a-a0aa-4a05ff9f3b24.png">
Figure 11: The predicted and the actual price of AMZN on 2021-12-27. If we look on top, [[176.20567]] is the predicted price from the model whereas 3393.389893 was the actual price on that day.

Because the program analyzed the stock data of Meta Platforms, Inc. and tried to predict the closing stock price of other companies, the results came out closer to the actual price and predictions for NVIDIA Corporation (NVDA) and Apple Inc. (AAPL) but was far off for the case of Amazon.com Inc.
