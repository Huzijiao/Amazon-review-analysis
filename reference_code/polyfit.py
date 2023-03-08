import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# synthesis score regression
def regression_s(microwave):
    data = pd.read_csv('total_score/' + microwave + ".csv")
    data_x = np.array(data['review_date'])
    data_y = np.array(data['total_score'])
    x_train, test_x, y_train, test_y = train_test_split(data_x, data_y, test_size=0.33, random_state=1)
    x = x_train.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y_train)
    test_x = test_x.reshape(-1, 1)
    weights = np.polyfit(x_train, y_train, 4)
    ans = np.poly1d(weights)
    print("y=", ans)
    model = np.poly1d(weights)
    xp = np.linspace(test_x.min(), test_x.max(), 70)
    pred_plot = model(xp)
    pred = model(test_x)
    plt.scatter(test_x, test_y, facecolor='blue', edgecolor='blue')
    plt.plot(xp, pred_plot, color='red')
    plt.title("synthesis score regression")
    plt.ylabel(product_name + ' synthesis score')
    plt.savefig('total_score/' + product_name + '.png', dpi=200, bbox_inches='tight')
    plt.show()
    MAE = mean_squared_error(test_y, pred.flatten())
    MSE = mean_absolute_error(test_y, pred.flatten())
    print("MSE", MSE)
    print("MAE", MAE)
if __name__ == '__main__':
    regression_s("hair_dryer")
