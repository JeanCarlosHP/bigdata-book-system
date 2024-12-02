import pandas as pd

file_path = './books.csv'
data = pd.read_csv(file_path)

data.head(), data.info()