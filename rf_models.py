import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


def train_rf(x_train, y_train, n_estimators=100, max_features=1, n_jobs=-1):
    rf = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs)
    rf.fit(x_train, y_train)
    return rf


def calculate_error(values_pr, values_true):
    return np.mean((values_pr - values_true) ** 2)


def train_models(data_x, data_y, env_name, samples_n_start=200, samples_n_period=200, plot=False):
    err = []
    err_train = []
    model = None
    for i in range(samples_n_start, len(data_x), samples_n_period):
        x_train, x_test, y_train, y_test = train_test_split(data_x[:i], data_y[:i], test_size=0.2)

        if env_name == 'Pendulum-v2':
            model = train_rf(x_train, y_train, max_features=3, n_estimators=300)
        elif env_name == 'Qube-v0':
            model = train_rf(x_train, y_train, max_features=5, n_estimators=800)
        elif env_name == 'Pendulum-v0':
            model = train_rf(x_train, y_train, max_features=3, n_estimators=500)
        else:
            raise EnvironmentError('No support for environment: {}'.format(env_name))

        predictions = model.predict(x_test)
        err.append(calculate_error(predictions, y_test))
        if plot:
            predictions_train = model.predict(x_train)
            err_train.append(calculate_error(predictions_train, y_train))

    if plot:
        plt.figure('RF model error')
        plt.xlabel('Samples number')
        plt.ylabel('Error')
        plt.plot(range(samples_n_start, len(data_x), samples_n_period), err, '-b', label='test error')
        plt.plot(range(samples_n_start, len(data_x), samples_n_period), err_train, '-r', label='train error')
        plt.show()

    return model, err
