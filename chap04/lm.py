import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

d = pd.read_csv("input/data-salary.txt")
x = sm.add_constant(d.X)
res_lm = sm.OLS(d.Y, x).fit()

new_x = sm.add_constant(range(23, 61))
summary = res_lm.get_prediction(new_x).summary_frame(alpha=0.05)