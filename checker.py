import pandas as pd
import numpy as np
import spacy
df=pd.read_json('Sarcasm_Headlines_Dataset.json',lines=True)
print(df.head())
