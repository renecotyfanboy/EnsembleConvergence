import sys
import numpy as np
import numpyro
import arviz as az
from jax import random
from model import eight_schools
from numpyro.infer import AIES, ESS, MCMC

if __name__ == "__main__":

    i = int(sys.argv[1])
    num_walkers = int(sys.argv[2])

    keys = ["aies", "ess", "aies_corrected", "ess_corrected"]
    diagnoses = {}
    for key in keys:
        for diag in ["rhat", "ess"]:

            diagnoses.update({f'{diag}_{key}': [] for key in keys})



    id_list = [
        az.from_netcdf(f"results/inference_data_{key}_{i}_{num_walkers}_walkers.nc") for key in keys
    ]

    for key, inference_data in zip(keys, id_list):

        rhat = az.rhat(inference_data)
        diagnoses[f"rhat_{key}"].append(np.concatenate([np.asarray(rhat.mu)[None], np.asarray(rhat.tau)[None], np.asarray(rhat.theta)]))

        ess = az.ess(inference_data)
        diagnoses[f"ess_{key}"].append(np.concatenate([np.asarray(ess.mu)[None], np.asarray(ess.tau)[None], np.asarray(ess.theta)]))

    np.savez(f"reduced_stats/diagnoses_{i}_{num_walkers}_walkers.npz", **diagnoses)