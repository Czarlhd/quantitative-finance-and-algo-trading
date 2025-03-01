import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

RISK_FREE_RATE = 0.05
MONTHS_IN_YEAR = 12


class CAPM:
    def __init__(self, stocks, start_date, end_date):
        self.data = None
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self):
        data = {}

        for stock in self.stocks:
            ticker = yf.Ticker(stock)
            data[stock] = ticker.history(start=self.start_date, end=self.end_date)[
                "Close"
            ]

        return pd.DataFrame(data)

    def initialize(self):
        stock_data = self.download_data()
        stock_data = stock_data.resample("ME").last()

        self.data = pd.DataFrame(
            {
                "s_close": stock_data[self.stocks[0]],
                "m_close": stock_data[self.stocks[1]],
            }
        )

        # Log monthly returns
        self.data[["s_returns", "m_returns"]] = np.log(
            self.data[["s_close", "m_close"]]
            / self.data[["s_close", "m_close"]].shift(1)
        )

        # Remove NaN values
        self.data = self.data[1:]

    def calculate_beta(self):
        # Covariance matric: the diagonal items are the variances
        # Off diagonals are the covariances
        # The matrix is symmetric: cov[0,1] = cov[1,0]
        covariance_matrix = np.cov(self.data["s_returns"], self.data["m_returns"])
        # Calculating the Beta
        beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
        print("Beta from formula: ", beta)

    def regression(self):
        # Using linear Regression to fit a line to the data
        # [stock_returns, market_returns] = slope is the beta
        beta, alpha = np.polyfit(self.data["m_returns"], self.data["s_returns"], deg=1)
        print("Beta from regression: ", beta)

        expected_return = RISK_FREE_RATE + beta * (
            self.data["m_returns"].mean() * MONTHS_IN_YEAR - RISK_FREE_RATE
        )
        print("Expected Return: ", expected_return)

        self.plot_regression(alpha, beta)

    def plot_regression(self, alpha, beta):
        fig, axis = plt.subplots(1, figsize=(20, 10))
        axis.scatter(
            self.data["m_returns"], self.data["s_returns"], label="Data Points"
        )
        axis.plot(
            self.data["m_returns"],
            beta * self.data["m_returns"] + alpha,
            color="red",
            label="CAPM Line",
        )
        plt.title("Capital Asset Pricing Model, finding alphas and betas")
        plt.xlabel("Market return $R_m$", fontsize=18)
        plt.ylabel("Stock return $R_a$")
        plt.text(0.08, 0.05, r"$R_a = \beta * R_m + \alpha$", fontsize=18)
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    capm = CAPM(["IBM", "^GSPC"], "2009-12-01", "2017-01-01")
    capm.initialize()
    capm.calculate_beta()
    capm.regression()
