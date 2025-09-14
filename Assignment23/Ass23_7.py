import pandas as pd
import matplotlib.pyplot as plt
data={
    'Name':['Amit','Sagar','Pooja'],
    'Math':[85,90,78],
    'Science':[92,88,80],
    'English':[75,85,82]
}
df=pd.DataFrame(data)

df['Total']=df['Math']+df['Science']+df['English']

plt.bar(df['Name'],df['Total'])
plt.xlabel('Student Name')
plt.ylabel('Total Marks')
plt.title('Bar Plot')
plt.grid(True)
plt.show()
