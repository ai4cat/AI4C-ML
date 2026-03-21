import torch, sys, math, scipy, random, json, xlrd, pandas, copy
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, TensorDataset
import xlrd
from sklearn.preprocessing import StandardScaler


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class AADataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.data_x = copy.deepcopy(dataset['data_x'])
        self.data_y = copy.deepcopy(dataset['data_y'])

        self.names = copy.deepcopy(dataset['names'])

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx], self.names[idx]

    def __len__(self):
        return len(self.data_x)


class DataProcessor():
    def __init__(self, seed=0) -> None:
        self.data_file = '/home/a/PKU/PKU240918/code_data/task_data/base_model_data_Ni1_Mn1_Ir1_Ni2_Ti1_Fe1_Cu1_Ni3.xlsx'
        self.labeled_data_sheet = 'DATA'
        self.data_x = []
        self.data_y = []
        self.names = []
        self.gp_seed = seed
        self.read_labeled_data()

    def read_labeled_data(self):
        # Panda 解析
        data_array = pandas.read_excel(
            self.data_file, sheet_name=self.labeled_data_sheet)

        for row_index in range(1, data_array.shape[0]):
            data = data_array.values[row_index]

            name = data[2]
            y = float(data[5])

            raw_x = [float(item) for item in data[6:26]]
            # xgboost
            self.data_x.append(list(raw_x))
            self.data_y.append(y)
            self.names.append(name)
        print("raw self.data_x.shape:", len(self.data_x), len(self.data_x[0]))

    def get_dataset(self):
        return {
            'data_x': np.array(self.data_x),
            'data_y': np.array(self.data_y).reshape(-1, 1),
            'names': np.array(self.names).reshape(-1, 1),
            'size': len(self.data_x)
        }

    @staticmethod
    def shuffle(dataset):
        # print("raw_data: ",dataset)
        indices = [i for i in range(len(dataset['data_x']))]
        random.shuffle(indices)
        # print("indices: ",indices)
        dataset['data_x'] = dataset['data_x'][indices]
        dataset['data_y'] = dataset['data_y'][indices]
        dataset['names'] = dataset['names'][indices]
        # print("shuflle_data: ",dataset)

        return dataset

    def split(dataset, j):
        indices = [i for i in range(len(dataset['data_x']))]
        random.shuffle(indices)
        # print("indices: ",indices)
        data1 = {}
        data2 = {}
        data1['data_x'] = dataset['data_x'][0:j]
        data1['data_y'] = dataset['data_y'][0:j]
        data1['names'] = dataset['names'][0:j]
        data1['size'] = len(data1['data_x'])
        data2['data_x'] = dataset['data_x'][j:]
        data2['data_y'] = dataset['data_y'][j:]
        data2['names'] = dataset['names'][j:]
        data2['size'] = len(data2['data_x'])
        return data1, data2

    def loo_validation(dataset, idx):
        val_set = {'data_x': [dataset['data_x'][idx]], 'data_y': [
            dataset['data_y'][idx]], 'names': [dataset['names'][idx]], 'size': 1}
        train_set = copy.deepcopy(dataset)
        train_set['data_x'] = np.delete(train_set['data_x'], idx, axis=0)
        train_set['data_y'] = np.delete(train_set['data_y'], idx, axis=0)
        train_set['names'] = np.delete(train_set['names'], idx, axis=0)
        train_set['size'] = train_set['size'] - 1

        return train_set, val_set


if __name__ == "__main__":
    AADataset(DataProcessor().get_dataset())