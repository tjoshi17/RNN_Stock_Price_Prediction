# 📈 Stock Price Prediction Using Recurrent Neural Networks (RNN, LSTM, GRU)

## Objective
The objective of this assignment is to try and predict the stock prices using historical data from four companies IBM (IBM), Google (GOOGL), Amazon (AMZN), and Microsoft (MSFT).

We use four different companies because they belong to the same sector: Technology. Using data from all four companies may improve the performance of the model. This way, we can capture the broader market sentiment.

The problem statement for this assignment can be summarised as follows:

- Given the stock prices of Amazon, Google, IBM, and Microsoft for a set number of days, predict the stock price of these companies after that window.

## Business Value
Data related to stock markets lends itself well to modeling using RNNs due to its sequential nature. We can keep track of opening prices, closing prices, highest prices, and so on for a long period of time as these values are generated every working day. The patterns observed in this data can then be used to predict the future direction in which stock prices are expected to move. Analyzing this data can be interesting in itself, but it also has a financial incentive as accurate predictions can lead to massive profits.

## Data Description
You have been provided with four CSV files corresponding to four stocks: AMZN, GOOGL, IBM, and MSFT. The files contain historical data that were gathered from the websites of the stock markets where these companies are listed: NYSE and NASDAQ. The columns in all four files are identical. Let's take a look at them:

- `Date`: The values in this column specify the date on which the values were recorded. In all four files, the dates range from Jaunary 1, 2006 to January 1, 2018.

- `Open`: The values in this column specify the stock price on a given date when the stock market opens.

- `High`: The values in this column specify the highest stock price achieved by a stock on a given date.

- `Low`: The values in this column specify the lowest stock price achieved by a stock on a given date.

- `Close`: The values in this column specify the stock price on a given date when the stock market closes.

- `Volume`: The values in this column specify the total number of shares traded on a given date.

- `Name`: This column gives the official name of the stock as used in the stock market.

There are 3019 records in each data set. The file names are of the format \<company_name>_stock_data.csv.

## 🧹 Data Preprocessing & Merging
- Datasets from four different companies — **Amazon (AMZN), Google (GOOGL), IBM (IBM),** and **Microsoft (MSFT)** — were merged into a single DataFrame.
- Data spans from **2006 to 2018**, with columns like `Open`, `High`, `Low`, `Close`, `Volume`, and `Name`.
- Employed **sliding window techniques** to create time-series samples for supervised learning.
- Normalized the data using `StandardScaler` to ensure different price ranges and volumes across stocks did not bias the model.

  
## 🛠️ Model Development
- A **Simple Recurrent Neural Network (RNN)** model was built to benchmark performance.
- Also developed **Advanced Recurrent Neural Network models** using:
  - **Long Short-Term Memory (LSTM)**
  - **Gated Recurrent Unit (GRU)**
- Both simple and advanced RNN models were designed for:
  - **Single target variable** (e.g., predicting Close price)
  - **Multiple target variables** (e.g., predicting Open, High, Low, Close)
- The models used **multiple features** (`Open`, `High`, `Low`, `Close`, `Volume`) as predictors.

  
## 🔮 Prediction Strategy
- Predictions were made on test datasets using all developed models.
- Focus was on predicting the **`Close` price**, a critical indicator for financial decision-making.


## 📉 Performance Evaluation
- **Training and validation losses** were plotted across epochs, showing consistent learning without overfitting.
- **Visual comparisons** between actual vs predicted closing prices showed:
  1. The model captured overall **trends and seasonal patterns**.
  2. Some **short-term deviations** were present, but predictions remained **directionally accurate**.


## 📊 Performance Comparison

All models were evaluated using:
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R-squared (R²)**
- **Correlation between actual vs predicted**

| Metric                  | Simple RNN (Single Target)  | Advanced RNN (Single Target - GRU)  | Simple RNN (Multiple Target)  | Advanced RNN (Multiple Target - LSTM)  |
|-------------------------|-----------------------------|-------------------------------------|-------------------------------|----------------------------------------|
| **Units**               | 32                          | 64                                  | 128                           | 128                                    |
| **Dropout Rate**        | 0.2                         | 0.1                                 | 0.2                           | 0.1                                    |
| **Learning Rate**       | 0.001                       | 0.0005                              | 0.001                         | 0.001                                  |
| **Activation Function** | ReLU                        | ReLU                                | ReLU                          | ReLU                                   |
| **MSE**                 | 0.0151                      | 0.0132                              | 0.0775                        | 0.0292                                 |
| **MAE**                 | 0.0942                      | 0.0934                              | 0.2232                        | 0.1278                                 |
| **R² Score**            | 0.9630                      | 0.9675                              | 0.7205                        | 0.8680                                 |
| **Corr (Actual vs Pred)**| 0.9845                     | 0.9846                              | 0.9591                        | 0.9828                                 |

 ✅ **Advanced RNN models (LSTM & GRU) significantly outperformed Simple RNNs**, especially in terms of lower prediction errors and higher R² scores, indicating superior generalization capability.


## ✅ Conclusion
- The **tuned GRU model** emerged as the **top performer**, delivering:
  1. **Accurate and robust** stock price forecasts
  2. **Minimal overfitting**, supported by dropout and early stopping
- **Hyperparameter tuning** (units, dropout, activation, learning rate) played a **crucial role** in maximizing performance.
- **Advanced models (GRU, LSTM)** consistently outperformed Simple RNNs, showcasing better long-term memory, learning efficiency, and predictive power.


## 💼 Business Insight
- The developed models provided **reliable short-term forecasts** for stock closing prices.
- Practical real-world applications include:
  1. **Risk Management**: Better estimation of potential financial losses.
  2. **Portfolio Optimization**: Improved allocation strategies.
  3. **Trading Signal Generation**: Enabling data-driven buy/sell decisions.
- By capturing both **trends and seasonality**, these models can be used as **decision-support tools** for quantitative analysts and financial strategists.

### Technologies
- numpy version: 1.26.4
- pandas version: 2.2.2
- seaborn version: 0.13.2
- matplotlib version: 3.10.0
- tensorflow version: 2.18.0
- keras version: 3.8.0
- sklearn version: 1.6.1

### Author
 > Tejashri Pilla
 >> **Contact** - tejashrii.joshi@gmail.com

