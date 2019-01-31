import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import pyro
from pyro import distributions as dist
from pyro.infer.mcmc import MCMC, NUTS
from pyro.infer.abstract_infer import EmpiricalMarginal
import torch.distributions as tdist



def model(data):
    alpha = 1 / torch.mean(data)
    lambda1 = pyro.sample("lambda1", dist.Exponential(alpha))
    lambda2 = pyro.sample("lambda2", dist.Exponential(alpha))
    tau = pyro.sample("tau", dist.Uniform(0, 1))
    idx = torch.arange(len(data)).float()
    lambda_ = torch.where(idx.lt(tau*len(data)), lambda1, lambda2)

    with pyro.plate("data", len(data)):
        pyro.sample("obs", dist.Poisson(lambda_), obs=data)


data = pd.read_csv("https://raw.githubusercontent.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/master/Chapter1_Introduction/data/txtdata.csv", header=None)

torch_data = torch.tensor(data.astype("float32")[0].values)

nuts_kernel = NUTS(model, adapt_step_size=True, jit_compile=True, ignore_jit_warnings=True)
hmc_posterior = MCMC(nuts_kernel, num_samples=1000, warmup_steps=100).run(torch_data)

data_mcmc = EmpiricalMarginal(hmc_posterior, ["lambda1", "lambda2", "tau"])._get_samples_and_weights()[0]

plt.figure()
sns.distplot(data_mcmc[:, 0])
sns.distplot(data_mcmc[:, 1])
plt.figure(figsize=(12.5, 5))
sns.distplot(data_mcmc[:, 2]*len(data))
plt.figure(figsize=(12.5, 5))
w = 1.0 / data_mcmc[:, 2].size(0) * torch.ones_like(data_mcmc[:, 2])
plt.hist(data_mcmc[:, 2], bins=len(data), alpha=1, weights=w, rwidth=2.)

plt.show()
