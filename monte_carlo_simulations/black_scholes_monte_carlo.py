import numpy as np


class OptionPricing:

    def __init__(self, S0, E, T, rf, sigma, iterations):
        self.SO = S0
        self.E = E
        self.T = T
        self.rf = rf
        self.sigma = sigma
        self.iterations = iterations

    def call_options_simulation(self):
        # we have 2 columns: first with 0s, the second column will store the payoff
        # we need the first column of 0s: payoff function is max(S-E, 0) for the call option
        option_data = np.zeros([self.iterations, 2])

        # dimenstions: 1 dimensional array with as manu items as the iterations
        rand = np.random.normal(0, 1, [1, self.iterations])

        # equation for the S(t) stock price at T
        stock_price = self.SO * np.exp(
            self.T * (self.rf - 0.5 * self.sigma**2)
            + self.sigma * np.sqrt(self.T) * rand
        )

        # we need the S-E because we have to calculate the max(S-E,0)
        option_data[:, 1] = stock_price - self.E

        # average for the Monte-Carlo simulation
        # max() returns the max(S-E, 0) according to the formula
        average = np.sum(np.amax(option_data, axis=1)) / float(self.iterations)

        # we have to use the exp(-rT) discount factor
        return np.exp(-1.0 * self.rf * self.T) * average

    def put_options_simulation(self):
        # we have 2 columns: first with 0s, the second column will store the payoff
        # we need the first column of 0s: payoff function is  max(E-S,0) for the call option
        option_data = np.zeros([self.iterations, 2])

        # dimenstions: 1 dimensional array with as manu items as the iterations
        rand = np.random.normal(0, 1, [1, self.iterations])

        # equation for the S(t) stock price at T
        stock_price = self.SO * np.exp(
            self.T * (self.rf - 0.5 * self.sigma**2)
            + self.sigma * np.sqrt(self.T) * rand
        )

        # we need the S-E because we have to calculate the max(E-S,0)
        option_data[:, 1] = self.E - stock_price

        # average for the Monte-Carlo simulation
        # max() returns the  max(E-S,0) according to the formula
        average = np.sum(np.amax(option_data, axis=1)) / float(self.iterations)

        # we have to use the exp(-rT) discount factor
        return np.exp(-1.0 * self.rf * self.T) * average


if __name__ == "__main__":
    model = OptionPricing(100, 100, 1, 0.05, 0.2, 10000)
    print("Value of the call option is $%.2f " % model.call_options_simulation())
    print("Value of the put option is $%.2f " % model.put_options_simulation())
