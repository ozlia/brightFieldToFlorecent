


# for i,(j,k) in enumerate([([1,2],[3,4])]):
#     print(i)
#     print(j)
#     print(k)

import pandas as pd

df = pd.DataFrame(columns=['first_col','second_col'])
new_row = dict(zip(df.columns,[1,2]))
df.loc[df.shape[0]] = new_row
df.loc[df.shape[0]] = new_row
print('?')