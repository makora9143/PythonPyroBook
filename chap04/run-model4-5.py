import torch
import pandas as pd

import pyro
from pyro.infer.mcmc import MCMC, NUTS
from pyro.infer.abstract_infer import EmpiricalMarginal
from model4_5 import model

pyro.set_rng_seed(1234)
# Enable validation checks
pyro.enable_validation(True)

d = pd.read_csv('input/data-salary.txt').astype('float32')

d = torch.tensor(d.values)
x_data, y_data = d[:, 0], d[:, -1]

nuts_kernel = NUTS(model, adapt_step_size=True, jit_compile=True, ignore_jit_warnings=True)
hmc_posterior = MCMC(nuts_kernel, num_samples=1000, num_chains=4, warmup_steps=200).run(x_data, y_data)

posterior_a = EmpiricalMarginal(hmc_posterior, 'a')
posterior_b = EmpiricalMarginal(hmc_posterior, 'b')
posterior_sigma = EmpiricalMarginal(hmc_posterior, 'sigma')
