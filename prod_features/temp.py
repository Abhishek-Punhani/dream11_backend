import pandas as pd

df=pd.read_csv(r'/home/manav/dev_ws/src/dream11_backend/prod_features/data/full_dataset.csv')

from model.train_model import train_model

train_model(df)