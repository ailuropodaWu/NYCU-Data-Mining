import pandas as pd
def save2csv(score, path):
    pd.DataFrame(score, columns=['outliers']).to_csv(path, index_label='id')