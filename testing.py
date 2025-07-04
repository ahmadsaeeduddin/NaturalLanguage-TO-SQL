import pandas as pd
from models.prompt_builder import build_prompt

dataset = pd.read_csv('models/spider_text_sql.csv')
print(dataset.head())

