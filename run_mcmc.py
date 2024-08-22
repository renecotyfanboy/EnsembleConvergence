import sys
import numpy as np
import numpyro
import arviz as az
from jax import random
from model import eight_schools
from numpyro.infer import AIES, ESS, MCMC

if __name__ == "__main__":

    numpyro.enable_x64()
    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(4)

    i = int(sys.argv[1])
    num_walkers = int(sys.argv[2])

    aies_kernel = AIES(eight_schools)
    mcmc = MCMC(aies_kernel, num_warmup=0, num_samples=10_000, num_chains=num_walkers, chain_method="vectorized", progress_bar=False)
    mcmc.run(random.PRNGKey(np.random.randint(0, int(1e6))))
    az.from_numpyro(mcmc).to_netcdf(f"results/inference_data_aies_{i}_{num_walkers}_walkers.nc")

    ess_kernel = ESS(eight_schools)
    mcmc = MCMC(ess_kernel, num_warmup=0, num_samples=10_000, num_chains=num_walkers, chain_method="vectorized", progress_bar=False)
    mcmc.run(random.PRNGKey(np.random.randint(0, int(1e6))))
    az.from_numpyro(mcmc).to_netcdf(f"results/inference_data_ess_{i}_{num_walkers}_walkers.nc")
