# LSTM - Predict Google Stock Price

-also:(Kaggle : https://www.kaggle.com/code/amirsoleymani/lstm-predict-google-stock-price )

This project demonstrates the application of Long Short-Term Memory (LSTM) networks to predict Google's stock prices. LSTMs, a type of recurrent neural network (RNN), are particularly well-suited for time series forecasting due to their ability to capture long-term dependencies in sequential data.

## Table of Contents

- [Introduction](#introduction)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

## Introduction

Predicting stock prices is a challenging task due to the volatile nature of financial markets. However, with the advent of deep learning techniques, especially LSTMs, it has become possible to model and predict stock price movements with reasonable accuracy. This project focuses on using LSTM networks to predict Google's stock prices by leveraging historical stock data.

## Data Collection

The dataset used in this project comprises historical stock data for Google, including features such as 'Open', 'Close', 'High', 'Low', and 'Volume'. This data can be obtained from financial data providers like Yahoo Finance or Alpha Vantage. For this project, we utilized the Yahoo Finance API to fetch the data.

## Data Preprocessing

Before feeding the data into the LSTM model, several preprocessing steps are performed:

1. **Normalization**: The stock data is normalized using `MinMaxScaler` from the `scikit-learn` library, scaling the data between 0 and 1. This step is crucial for improving the convergence of the model during training.

2. **Window Sliding Method**: To create sequences of past data (lookback period) for predicting future stock prices, we employ the window sliding method. This approach structures the data into input-output pairs suitable for time series forecasting.

## Model Architecture

The LSTM model is built using the Keras library and follows a sequential architecture:

- **Input Layer**: Accepts the input sequences.
- **LSTM Layers**: Multiple LSTM layers with a specified number of units to capture temporal dependencies.
- **Dropout Layers**: Added after each LSTM layer to prevent overfitting by randomly setting a fraction of input units to 0 during training.
- **Dense Layer**: A fully connected layer that serves as the output layer, providing the predicted stock price.

## Training the Model

The model is compiled using the 'mean_squared_error' loss function and the 'adam' optimizer. Training is conducted over a specified number of epochs with a defined batch size. The training process involves:

- Forward propagation through the network.
- Backpropagation to update the weights based on the loss.
- Validation against a separate validation set to monitor performance and prevent overfitting.

## Evaluation

After training, the model's performance is evaluated using metrics such as Mean Squared Error (MSE) and Root Mean Squared Error (RMSE). Additionally, visualizations are created to compare the predicted stock prices against the actual prices, providing insights into the model's accuracy.

## Usage

To replicate this project:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/AmirHosseinSoleymani/LSTM-Predict-Google-Stock-Price.git
   cd LSTM-Predict-Google-Stock-Price
   ```

2. **Install Required Libraries**:

   Ensure you have the following Python libraries installed:

   - `numpy`
   - `pandas`
   - `matplotlib`
   - `scikit-learn`
   - `keras`
   - `yfinance`

   You can install them using `pip`:

   ```bash
   pip install numpy pandas matplotlib scikit-learn keras yfinance
   ```

3. **Run the Jupyter Notebook**:

   Open and execute the `lstm-predict-google-stock-price.ipynb` notebook to train the model and make predictions.

## Results

The model's predictions are visualized alongside the actual stock prices to assess performance. The visualizations indicate the model's capability to capture the general trend of the stock prices, with some deviations during highly volatile periods.

## Conclusion

This project illustrates the effectiveness of LSTM networks in predicting stock prices. While the model captures the overall trend, it's important to note that stock price prediction remains inherently uncertain due to market volatility. Further improvements can be achieved by incorporating additional features, tuning hyperparameters, and exploring more advanced architectures.

## License

This project is licensed under the MIT License.

---

Feel free to customize this README further to align with your project's specifics and any additional insights you wish to convey. 
