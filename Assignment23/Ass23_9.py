import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data={
    'Name':['Amit','Sagar','Pooja'],
    'Math':[np.nan,90,78],
    'Science':[92,np.nan,80]
}
df=pd.DataFrame(data)

df=df.fillna(df.mean(numeric_only=True))

print(df)