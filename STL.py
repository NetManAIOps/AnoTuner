import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.datasets import co2



def STLDecopose(x, season):
    result = stl.fit(x, season)

    return result.trend, result.seasonal


if __name__ == '__main__':
    data = co2.load_pandas().data
    data['co2'].interpolate(method='linear', inplace=True)

    # STL分解
    stl = STL(data['co2'], seasonal=13)
    result = stl.fit()

    # 绘图展示结果
    fig, axes = plt.subplots(4, figsize=(8, 6))

    axes[0].plot(data['co2'], label='Original')
    axes[0].legend(loc='upper left')

    axes[1].plot(result.trend, label='Trend')
    axes[1].legend(loc='upper left')

    axes[2].plot(result.seasonal, label='Seasonal')
    axes[2].legend(loc='upper left')

    axes[3].plot(result.resid, label='Residual')
    axes[3].legend(loc='upper left')

    plt.tight_layout()
    plt.show()
