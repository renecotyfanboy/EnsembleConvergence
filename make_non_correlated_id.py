import sys
import numpy as np
import numpyro
import arviz as az
import random
from model import eight_schools
from numpyro.infer import AIES, ESS, MCMC

if __name__ == "__main__":

    j = int(sys.argv[1])
    num_walkers = int(sys.argv[2])
    total_runs = 100

    inference_data_aies_list = [az.from_netcdf(f"results/inference_data_aies_{i}_{num_walkers}_walkers.nc") for i in range(total_runs)]
    inference_data_ess_list = [az.from_netcdf(f"results/inference_data_ess_{i}_{num_walkers}_walkers.nc") for i in range(total_runs)]

    az.concat(
        [
            inference_data_aies_list[i].isel(chain=[np.random.randint(0, num_walkers)])
            for i in random.sample(range(total_runs), num_walkers)
        ], dim="chain"
    ).to_netcdf(f"results/inference_data_aies_corrected_{j}_{num_walkers}_walkers.nc")

    az.concat(
        [
            inference_data_ess_list[i].isel(chain=[np.random.randint(0, num_walkers)])
            for i in random.sample(range(total_runs), num_walkers)
        ], dim="chain"
    ).to_netcdf(f"results/inference_data_ess_corrected_{j}_{num_walkers}_walkers.nc")