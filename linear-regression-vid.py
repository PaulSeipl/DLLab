import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
print(torch.__version__)

def create_dataset(sample_size=10, sigma=0.1, w_star=1, b_star = 1,
                   x_range=(-1, 1), seed=0):
    # Set the random state in numpy
    torch.manual_seed(seed)
    # Unpack the values in x_range
    x_min, x_max = x_range
    # Sample sample_size points from a uniform distribution
    X = torch.rand(sample_size)
    # Rescale between x_min and x_max 
    X = X * (x_max - x_min) + x_min
    # Compute hat(y)
    y_hat = X * w_star + b_star
    # Compute y (Add Gaussian noise)
    y = y_hat + torch.normal(torch.zeros(sample_size), sigma*torch.ones(sample_size))
    return X, y


def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].studytime
        y = points.iloc[i].score
        total_error += (y - (m * x + b)) ** 2
        
    return total_error / float(len(points))

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    
    n = len(points)
    
    for i in range(n):
        x = points.iloc[i].studytime
        y = points.iloc[i].score
        
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
        
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

def linear_regression_main():
    X, y = create_dataset(100, w_star=10, b_star=-3, sigma=100, x_range=(-20,80))
    t_data = np.array([X.numpy(), y.numpy()]).T
    data = pd.DataFrame(t_data, columns=['studytime', 'score'])
    print(int(data["studytime"].min()))

    m = 0
    b = 0
    L = 0.0001
    epochs = 300

    for i in range(epochs):
        if i % 50 == 0:
            print(f"Epoch: {i}")
        m, b = gradient_descent(m, b, data, L)
        
    print(m, b)

    plt.scatter(data.studytime, data.score, color="black", alpha=0.5)
    xs = range(int(data["studytime"].min()), int(data["studytime"].max()) + 1)
    plt.plot(list(xs), [m * x + b for x in xs], color="red")
    plt.show()

# Polynomial
def poly_main():
    np.random.seed(42)
    X = 4* np.random.rand(100, 1) - 2
    y = 4 + 2 * X + 5 * X**2 + 12 * X ** 3 + 2 * X ** 4 + 30 * np.random.randn(100, 1)

    poly_features = PolynomialFeatures(degree=4, include_bias=False)
    X_poly = poly_features.fit_transform(X)

    reg = LinearRegression()
    reg.fit(X_poly, y)

    X_vals = np.linspace(-2, 2, 100).reshape(-1, 1)
    X_vals_poly = poly_features.transform(X_vals)

    y_vals = reg.predict(X_vals_poly)
    print(X_vals_poly[:5])

    plt.scatter(X, y)
    plt.plot(X_vals, y_vals, color="r")
    plt.show()

   
