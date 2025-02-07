# LSTM - Predict Google Stock Price
-also:(Kaggle : https://www.kaggle.com/code/amirsoleymani/lstm-predict-google-stock-price )

This project demonstrates the use of LSTM (Long Short-Term Memory) networks to predict Google stock prices. LSTM, a type of recurrent neural network (RNN), is ideal for time series forecasting due to its ability to capture long-term dependencies in data.

The model is trained on historical stock data, including features like 'Open', 'Close', 'High', 'Low', and 'Volume'. The data is preprocessed using **MinMaxScaler**, which scales the data between 0 and 1, improving model convergence.

The **window sliding method** is employed to create sequences of past data (lookback period) to predict future stock prices. A sliding window helps structure data for time series forecasting.

## Key Features:
- **LSTM Model**: Used for time series prediction, capturing patterns over time.
- **MinMaxScaler**: Normalizes the stock data to a range between 0 and 1.
- **Window Sliding Method**: Converts historical data into sequential inputs for the LSTM model.

## Importing Required Libraries

To build and train our LSTM model, we need several essential Python libraries:

- **NumPy**: Provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions.
- **Pandas**: Used for data manipulation and analysis, particularly useful for handling time-series stock market data.
- **Matplotlib**: A visualization library for creating plots, which helps in analyzing and comparing stock prices.
- **Scikit-learn (MinMaxScaler)**: Used for normalizing the dataset to bring all values between a specific range (0 to 1) to improve model performance.
- **Keras (Sequential, LSTM, Dropout, Dense)**:
  - `Sequential`: A linear stack of layers for building deep learning models.
  - `LSTM`: Long Short-Term Memory, a specialized form of recurrent neural networks (RNN) designed for sequential data.
  - `Dropout`: A regularization technique to prevent overfitting by randomly ignoring some neurons during training.
  - `Dense`: A fully connected layer that serves as the output layer in our LSTM model.

The following code imports all the necessary libraries:


## License:
This project is licensed under the MIT License.

---
