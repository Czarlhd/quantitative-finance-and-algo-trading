import numpy as np
import yfinance as yf
import pandas as pd
import datetime


def download_data(stock, start_date, end_date):
    data = {}
    ticker = yf.Ticker(stock)
    data["Close"] = ticker.history(start=start_date, end=end_date)["Close"]
    return pd.DataFrame(data)


class ValueAtRiskMonteCarlo:

    def __init__(self, S, mu, sigma, c, n, iterations):
        # This is the value of our investement at t=0
        self.S = S
        self.mu = mu
        self.sigma = sigma
        self.c = c
        self.n = n
        self.iterations = iterations

    def simulation(self):
        rand = np.random.normal(0, 1, [1, self.iterations])

        # Equation for the S(t) stock price
        # The random walk of our initial investement
        stock_price = self.S * np.exp(
            self.n * (self.mu - 0.5 * self.sigma**2)
            + self.sigma * np.sqrt(self.n) * rand
        )

        # We have to sort the stock prices
        stock_price = np.sort(stock_price)

        # It depends on the confidence level: 95% -> 5 and 99% -> 1
        percentile = np.percentile(stock_price, (1 - self.c) * 100)

        return self.S - percentile


if __name__ == "__main__":

    S = 1e6  # this is the initial investment
    c = 0.95  # Conrfifence level: 95%
    n = 1  # 1 day
    iterations = 100000  # Number of paths of the Monte-Carlo simulation

    # Historical data to approximate mean and standard deviation
    start_date = datetime.datetime(2014, 1, 1)
    end_date = datetime.datetime(2017, 10, 15)

    citi = download_data("C", start_date, end_date)
    citi["returns"] = citi["Close"].pct_change()

    # We can assume daily returns to be normally distributed: mean and variance (standard deviation)
    # can describe the process
    mu = np.mean(citi["returns"])
    sigma = np.std(citi["returns"])

    model = ValueAtRiskMonteCarlo(S, mu, sigma, c, n, iterations)
    print("Value at Risk with the Monte-Carlo simulation: $%0.2f" % model.simulation())
