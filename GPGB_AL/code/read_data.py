# ========================= DATA TABLE MATCHING RULES =========================
# This file reads the labeled training table only.  Do not assume that every
# future Excel file has the same column order or the same metadata columns.
#
# Current labeled-table contract:
#   - one header row, read by pandas.read_excel(..., header=0)
#   - "Database number" is the sample name/identifier
#   - "Y" is the supervised regression target
#   - "X0" ... "X19" are the 20 descriptor columns used by the model
#   - other columns (for example metal labels, X20+, energies, notes, or
#     empty separator columns) are not training features in this reader
#
# For every new dataset, first inspect the header and explicitly update the
# named-column mapping below.  Do not replace it with a positional slice such
# as data[6:26].  A different file may place Y, the ID, or the descriptors at
# different positions, and silently reading the wrong columns changes the
# scientific meaning of the model without necessarily raising an error.
# If the new table has a different target, ID, or feature definition, record
# that mapping in the code and validate the required columns before training.
# ============================================================================

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
        return self.data_x[idx], self.data_y[idx], str(self.names[idx][0])

    def __len__(self):
        return len(self.data_x)


class DataProcessor():
    def __init__(self, seed=0) -> None:
        # This is a machine-specific example path. Before running, confirm and
        # update it to the local GPGB labeled workbook. The sheet and named
        # columns below must be checked against that local file as well.
        self.data_file = '/media/sf_Projects/ORR/GPGB/data/2_base_model_data_Ni1.xlsx'
        self.labeled_data_sheet = 'DATA'
        self.data_x = []
        self.data_y = []
        self.names = []
        self.gp_seed = seed
        self.read_labeled_data()

    def read_labeled_data(self):
        # This reader is intentionally restricted to the labeled training
        # table. Candidate/prediction tables are handled separately in
        # main.py because their metadata and column layout may be different.
        # Before using another file, verify its header and adjust the mapping
        # below rather than assuming that the columns are in the same order.
        data_array = pandas.read_excel(
            self.data_file, sheet_name=self.labeled_data_sheet)

        # Keep the feature definition explicit. If a future dataset uses a
        # different descriptor set or names, update this list deliberately and
        # re-check the model input dimension before running any training.
        feature_cols = [f'X{i}' for i in range(20)]
        required_cols = ['Database number', 'Y'] + feature_cols
        missing_cols = [col for col in required_cols if col not in data_array.columns]
        if missing_cols:
            raise ValueError(
                'Missing required DATA columns: {}'.format(', '.join(missing_cols))
            )

        labeled_data = data_array[required_cols]
        if labeled_data.isnull().any().any():
            missing_rows = labeled_data.index[labeled_data.isnull().any(axis=1)].tolist()
            raise ValueError(
                'Missing values found in required DATA columns at rows: {}'.format(
                    missing_rows
                )
            )

        for _, data in labeled_data.iterrows():
            name = str(data['Database number'])
            y = float(data['Y'])
            raw_x = [float(data[col]) for col in feature_cols]
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
        # indices = [i for i in range(len(dataset['data_x']))]
        # random.shuffle(indices)
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
