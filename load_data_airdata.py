from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot as plt

filename = 'airdata.csv'

def prase(x):
    return datetime.strptime(x, '%Y %m %d %H')
    #return datetime.strptime(x)

def load_dataset():
    # 导入数据
    #dataset = read_csv(filename, parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=prase)
    dataset = read_csv(filename, index_col=0)

    # 设定列名
    #dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    #dataset.index.name = 'date'

    # 使用中位数填充缺失值
    #dataset['pollution'].fillna(dataset['pollution'].mean(), inplace=True)

    return dataset

if __name__ == '__main__':
    dataset = load_dataset()
    print(dataset.head(5))

    # 查看数据的变化趋势
    #groups = [0, 1, 2, 3, 5, 6, 7]
    groups = range(dataset.shape[1])
    plt.figure()
    i = 1
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(dataset.values[:, group])
        plt.title(dataset.columns[group], y = 0.5, loc='right')
        i = i + 1
    plt.show()

