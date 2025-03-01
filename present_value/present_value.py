from math import exp


def future_descrete_value(x, r, n):
    return x * (1 + r) ** n


def present_descrete_value(x, r, n):
    return x / (1 + r) ** -n


def future_continuous_value(x, r, t):
    return x * exp(r * t)


def present_continuous_value(x, r, t):
    return x / exp(-r * t)


if __name__ == "__main__":

    # value of investment in $
    x = 100
    # interest rate
    r = 0.05
    # duration (year)
    n = 5

    print("Future Value (discrete model) of x: %s" % future_descrete_value(x, r, n))
    print("Present Value (discrete model) of x: %s" % present_descrete_value(x, r, n))
    print("Future Value (continuous model) of x: %s" % future_continuous_value(x, r, n))
    print(
        "Present Value (continuous model) of x: %s" % present_continuous_value(x, r, n)
    )
