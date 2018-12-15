import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


class OpenAIGymDataset(Dataset):

    def __init__(self, data_x, data_y):
        self.data_x = torch.from_numpy(data_x).float()
        self.data_y = torch.from_numpy(data_y).float()
        self.len = len(self.data_x)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.len


def create_nn_model(d_in, h, d_out, complex_model=0):
    loss_fn = nn.MSELoss()
    if complex_model:
        model = nn.Sequential(
            nn.Linear(d_in, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.Sigmoid(),
            nn.Linear(h, d_out)
        )
    else:
        model = nn.Sequential(
            nn.Linear(d_in, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Sigmoid(),
            nn.Linear(h, h),
            nn.Sigmoid(),
            nn.Linear(h, d_out)
        )
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    return model, loss_fn, optimizer


def get_data_loaders(data_x, data_y):
    x_train, x_test, y_train, y_test, = train_test_split(data_x, data_y, test_size=0.2)

    train_dataset = OpenAIGymDataset(x_train, y_train)
    test_dataset = OpenAIGymDataset(x_test, y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
    return train_loader, test_dataset, train_dataset


def train_models(data_x, data_y, env_name, samples_n_start=100, samples_n_period=100, n_epochs=300,
                 plot=False):
    indexes = np.arange(len(data_x))
    np.random.shuffle(indexes)
    data_x = data_x[indexes]
    data_y = data_y[indexes]

    err = []
    err_train = []
    model = None
    for i in range(samples_n_start, len(data_x), samples_n_period):
        if env_name == 'Pendulum-v0':
            model, loss_fn, optimizer = create_nn_model(4, 14, 4)
        elif env_name == 'Pendulum-v2':
            model, loss_fn, optimizer = create_nn_model(3, 12, 3)
        else:
            raise EnvironmentError('No support for environment: {}'.format(env_name))

        train_loader, test_dataset, train_dataset = get_data_loaders(data_x[:i], data_y[:i])

        for t in range(n_epochs):
            for j, batch in enumerate(train_loader):
                batch_x, batch_y = batch

                pred = model(batch_x)
                loss = loss_fn(pred, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # print(f'Epoch: {t + 1}')

        with torch.no_grad():
            test_x, test_y = test_dataset.data_x, test_dataset.data_y
            pred = model(test_x)
            loss = loss_fn(pred, test_y)
            err.append(loss)
            if plot:
                train_x, train_y = train_dataset.data_x, train_dataset.data_y
                pred_train = model(train_x)
                loss_tr = loss_fn(pred_train, train_y)
                err_train.append(loss_tr)

    if plot:
        plt.figure('NN model error')
        plt.xlabel('Samples number')
        plt.ylabel('Error')
        plt.plot(range(samples_n_start, len(data_x), samples_n_period), err, '-b', label='test error')
        plt.plot(range(samples_n_start, len(data_x), samples_n_period), err_train, '-r', label='test error')
        plt.show()

    return model, err
