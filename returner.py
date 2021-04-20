import pandas as pd
import numpy as np
datanew = pd.read_csv("tweetgot.csv")
arr = np.asarray(datanew["sentiment"])
s = " "
for i in arr:
    s = s + str(i)
