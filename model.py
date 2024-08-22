import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import LocScaleReparam


def eight_schools(observed=True):

    J = 8
    y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

    mu = numpyro.sample('mu', dist.Normal(0, 5))
    tau = numpyro.sample('tau', dist.LogNormal(np.log(5), 1))

    with numpyro.plate('J', J):
        with numpyro.handlers.reparam(config={'theta': LocScaleReparam(centered=0)}):
            theta = numpyro.sample('theta', dist.Normal(mu, tau))

        numpyro.sample('obs', dist.Normal(theta, sigma), obs=y if observed else None)