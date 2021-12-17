


# for i,(j,k) in enumerate([([1,2],[3,4])]):
#     print(i)
#     print(j)
#     print(k)

import pandas as pd

col_dict = {
    'first_col': [],
    'second_col' : []
}

for k, v in zip(col_dict.keys(), [1,2]):
    col_dict[k].append(v)

df = pd.DataFrame(col_dict)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)