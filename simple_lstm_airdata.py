import numpy as np
from pandas import DataFrame, concat
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.metrics.regression import r2_score, mean_squared_error

batch_size = 72
epochs = 2000
# 通过过去几次的数据进行预测
n_input = 24*7

def convert_dataset(data, n_input=1, out_index=0, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = [], []
    # 输入序列 (t-n, ... t-1)
    for i in range(n_input, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # 输出结果 (t)
    cols.append(df[df.columns[out_index]])
    names += ['result']
    # 合并输入输出序列
    result = concat(cols, axis=1)
    result.columns = names
    # 删除包含缺失值的行
    if dropnan:
        result.dropna(inplace=True)
    return result


# class_indexs 编码的字段序列号，或者序列号List，列号从0开始
def class_encode(data, class_indexs):
    encoder = LabelEncoder()
    class_indexs = class_indexs if type(class_indexs) is list else [class_indexs]

    values = DataFrame(data).values

    for index in class_indexs:
        values[:, index] = encoder.fit_transform(values[:, index])

    return DataFrame(values) if type(data) is DataFrame else values

def build_model(lstm_input_shape):
    model = Sequential()
    model.add(LSTM(units=50, input_shape=lstm_input_shape, return_sequences=True))
    model.add(LSTM(units=50, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    model.summary()
    return model

if __name__ == '__main__':
    # 导入数据
    data = pd.read_csv('airdata.csv', sep=',', dtype=np.float64)

    # 对风向列进行编码
    #data = class_encode(data, 4)
    # 生成数据集，使用前5次的数据，来预测新数据
    dataset = convert_dataset(data, n_input=n_input)
    values = dataset.values.astype('float32')

    # 分类训练与评估数据集
    features = data.drop('pm25', axis=1).values
    training_features, testing_features, training_target, testing_target = train_test_split(features, data['pm25'].values, random_state=None)

    # 数据归一元(0-1之间)
    scaler = MinMaxScaler()
    training_features = scaler.fit_transform(training_features)
    testing_features = scaler.fit_transform(testing_features)

    # 将数据整理成【样本，时间步长，特征】结构
    training_features = training_features.reshape(training_features.shape[0], 1, training_features.shape[1])
    testing_features = testing_features.reshape(testing_features.shape[0], 1, testing_features.shape[1])
    # 查看数据维度
    print(training_features.shape, testing_features.shape, training_target.shape, testing_target.shape)

    # 训练模型
    lstm_input_shape = (training_features.shape[1], training_features.shape[2])
    model = build_model(lstm_input_shape)
    model.fit(training_features, training_target, batch_size=batch_size, validation_data=(testing_features, testing_target), epochs=epochs, verbose=2)

    # 使用模型预测评估数据集
    predictions = model.predict(testing_features)

    print('Mean Absolute Error = %0.4f' % np.mean(abs(predictions - testing_target)))
    print('R-squared:%0.4f MSE:%0.4f' % (r2_score(testing_target, predictions), mean_squared_error(testing_target, predictions)))

    # 图表显示
    plt.plot(testing_target, color='blue', label='Actual')
    plt.plot(predictions, color='green', label='Prediction')
    plt.legend(loc='upper right')
    plt.show()