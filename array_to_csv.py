import numpy as np
import pandas as pd

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
columns = ["test", "test2", "test3"]

df = pd.DataFrame(data, columns=columns)
df.to_csv('data.csv', index=False)
