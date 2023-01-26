import random
import numpy as np


def getRandomSeed():
    random.seed()
    seed = random.randint(0, 2 ** 32)
    return seed


def resetAllSeeds(master_seed=None):
    if master_seed is None:
        master_seed = getRandomSeed()

    msd = master_seed + 1
    n_vp = nest.GetKernelStatus(["total_num_virtual_procs"])[
        0
    ]  # number of virtual processes

    np_seed = master_seed
    py_seeds = range(msd, msd + n_vp)
    grng_seed = msd + n_vp
    rng_seeds = range(msd + n_vp + 1, msd + 2 * n_vp + 1)

    resetNumpySeed(np_seed)
    resetPythonSeeds(py_seeds)
    resetNestSeeds(grng_seed, rng_seeds)
    return {"master_seed": master_seed, "msd": msd, "n_vp": n_vp}


def resetNumpySeed(np_seed):
    np.random.seed(np_seed)
    return np_seed


# this is actually not resetting anything, but creating an rng for each VP
def resetPythonSeeds(py_seeds):
    pyrngs = [np.random.RandomState(s) for s in py_seeds]
    return py_seeds


def resetNestSeeds(grng_seed, rng_seeds):
    nest.SetKernelStatus({"grng_seed": grng_seed, "rng_seeds": rng_seeds})
    return grng_seed, rng_seeds
