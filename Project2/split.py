import pandas as pd
import os
from sklearn.model_selection import train_test_split

root = './dataset/'
os.makedirs(root, exist_ok=True)
output_train = os.path.join(root, 'train.json')
output_val = os.path.join(root, 'val.json')
input = './raw_dataset/train.json'

df = pd.read_json(input)

train, val = train_test_split(df, test_size=0.3, random_state=42)

train.to_json(output_train, orient='records')
val.to_json(output_val, orient='records')
