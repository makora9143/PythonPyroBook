import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm


d = pd.read_csv("input/data-salary.txt")
x = sm.add_constant(d.X)
res_lm = sm.OLS(d.Y, x).fit()

new_x = sm.add_constant(range(23, 61))
summary95 = res_lm.get_prediction(new_x).summary_frame(alpha=0.05)
summary50 = res_lm.get_prediction(new_x).summary_frame(alpha=0.5)


d.plot(kind="scatter", x="X", y="Y")
plt.plot(range(23, 61), summary95["mean"])
plt.fill_between(range(23, 61), summary95["mean_ci_lower"],summary95["mean_ci_upper"], color='blue', alpha=0.1)
plt.fill_between(range(23, 61), summary50["mean_ci_lower"],summary50["mean_ci_upper"],color='blue', alpha=0.3)

d.plot(kind="scatter", x="X", y="Y")
plt.plot(range(23, 61), summary95["mean"])
plt.fill_between(range(23, 61), summary95["obs_ci_lower"],summary95["obs_ci_upper"], color='blue', alpha=0.1)
plt.fill_between(range(23, 61), summary50["obs_ci_lower"],summary50["obs_ci_upper"],color='blue', alpha=0.3)

plt.show()
