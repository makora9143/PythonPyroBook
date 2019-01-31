import torch.nn as nn
import pyro
import pyro.distributions as dist


def model(x, y):
    a = pyro.sample("a", dist.Normal(0, 100))
    b = pyro.sample("b", dist.Normal(0, 100))

    sigma = pyro.sample("sigma", dist.Uniform(0., 100))

    with pyro.plate("data", x.size(0)):
        pyro.sample("obs", dist.Normal(a + b * x, sigma), obs=y)
