import matplotlib.pyplot as plt
import pandas as pd


d = pd.read_csv("input/data-salary.txt")
d.plot(kind="scatter", x="X", y="Y")
plt.show()
