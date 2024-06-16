import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

train_data = pd.read_csv('./train.csv')
raw_data = pd.read_csv('./raw_data.csv').sort_values(by='lettr', ignore_index=True)
test_data = pd.read_csv('./test_X.csv')
val_data = pd.concat([train_data, raw_data]).drop_duplicates(keep=False, ignore_index=True)
val_data.to_csv('./val.csv', index=False)

labels = []
for _, test_row in tqdm(test_data.iterrows(), total=len(test_data)):
    test_row = test_row.values
    for _, val_row in raw_data.iterrows():
        val_row = val_row.values
        label = val_row[0]
        if (test_row == val_row[1:]).all():
            labels.append(label)
            break
test_data['lettr'] = labels
temp = test_data.iloc[:, :-1]
test_data.iloc[:, 0] = test_data.iloc[:, -1]
test_data.iloc[:, 1:] = temp
test_data.to_csv('./test.csv', index=False, header=train_data.columns)
# print(train_data)
# print(val_data)
# print(test_data)
