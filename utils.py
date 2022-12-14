import pickle
import numpy as np
import pandas as pd


def calc_kld(generated_data, ground_truth, bins, range_min, range_max):
    pd_gt, _ = np.histogram(ground_truth, bins=bins, density=True, range=(range_min, range_max))
    pd_gen, _ = np.histogram(generated_data, bins=bins, density=True, range=(range_min, range_max))
    kld = 0
    for x1, x2 in zip(pd_gt, pd_gen):
        if x1 != 0 and x2 == 0:
            kld += x1
        elif x1 == 0 and x2 != 0:
            kld += x2
        elif x1 != 0 and x2 != 0:
            kld += x1 * np.log(x1 / x2)

    return np.abs(kld)


def prepare_dataset(dataset, condition_size=None):
    if dataset == "lorenz":
        with open("./datasets/lorenz/lorenz_dataset.pickle", "rb") as infile:
            dataset = pickle.load(infile)

        x_train = np.concatenate(list(dataset["x_train"].values()))
        y_train = np.concatenate(list(dataset["y_train"].values()))

        x_val = np.concatenate(list(dataset["x_val"].values()))
        y_val = np.concatenate(list(dataset["y_val"].values()))

        x_test = np.concatenate(list(dataset["x_test"].values()))
        y_test = np.concatenate(list(dataset["y_test"].values()))

    elif dataset == "mg":
        raw_dataset = pd.read_csv("./datasets/mg/MackyG17.csv")
        raw_dataset = np.transpose(raw_dataset.values)[0]

        x = [raw_dataset[i - condition_size:i] for i in range(condition_size, raw_dataset.shape[0])]
        x = np.array(x)
        y = raw_dataset[condition_size:]

        x_train = x[:int(x.shape[0] * 0.5)]
        y_train = y[:int(x.shape[0] * 0.5)]
        x_val = x[int(x.shape[0] * 0.5):int(x.shape[0] * 0.6)]
        y_val = y[int(x.shape[0] * 0.5):int(x.shape[0] * 0.6)]
        x_test = x[int(x.shape[0] * 0.6):]
        y_test = y[int(x.shape[0] * 0.6):]

    elif dataset == "itd":
        with open("./datasets/itd/a5m.pickle", "rb") as in_file:
            raw_dataset = pickle.load(in_file).astype(float)

        x = [raw_dataset[i - condition_size:i] for i in range(condition_size, raw_dataset.shape[0])]
        x = np.array(x)
        y = raw_dataset[condition_size:]

        x_train = x[:int(x.shape[0] * 0.5)]
        y_train = y[:int(x.shape[0] * 0.5)]
        x_val = x[int(x.shape[0] * 0.5):int(x.shape[0] * 0.6)]
        y_val = y[int(x.shape[0] * 0.5):int(x.shape[0] * 0.6)]
        x_test = x[int(x.shape[0] * 0.6):]
        y_test = y[int(x.shape[0] * 0.6):]

    return x_train, y_train, x_val, y_val, x_test, y_test
