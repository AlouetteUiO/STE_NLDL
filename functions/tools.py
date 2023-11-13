from scipy.stats import entropy
import numpy as np
from pytest import approx

def get_entropy(probability_distributon, axis=None):
    """
    Compute the entropy (information value) of the probability distribution of a random variable.

    The definition of Shannon's entropy S for a (discrete) random variable X reads:
    S = - SUM(p_j * ln(p_j))
    whereby X takes N possible values x_j (j = 1, ..., N) with probabilities p_j normalized
    to unity: SUM_j=1^N p_j = 1.
    S = 0 if and only if there is no uncertainty on X, for example one event has p=1 and all other events have p=0."
    
    base = 2, gives entropy in bits
    default is base = e, gives entropy in nats
    
    Parameters
    ----------
    probability distribution: array
        probability distribution of a random variable
    axis: scalar (int)
        axis of the probability distribution array

    Returns
    -------
    entropy: scalar (float)
        information value in bits of the random variable
    """
    return entropy(probability_distributon, axis=axis)


def get_DRPS(probability_distribution, pdf_true):
    """
    Get DRPS and relative DRPS
    discrete random probability score
    """
    Ndim = probability_distribution.ndim
    
    DRPS = np.zeros(Ndim)
    relDRPS = np.zeros(Ndim)

    for i in range(Ndim):

        a = list(range(Ndim))
        a.pop(i)

        # posterior distribution
        pdf = np.sum(probability_distribution, axis=tuple(a))
        assert np.sum(pdf) == approx(1)
        cdf = np.cumsum(pdf)

        # prior distribution
        pdf_prior = np.ones(len(pdf))/len(pdf)
        assert np.sum(pdf_prior) == approx(1)
        cdf_prior = np.cumsum(pdf_prior)

        # true distribution (1 on truth)
        cdf_true = np.cumsum(np.sum(pdf_true, axis=tuple(a)))

        DRPS_posterior = np.sum((cdf - cdf_true)**2)
        #print(DRPS_posterior)
        DRPS_prior = np.sum((cdf_prior - cdf_true)**2)
        #print(DRPS_prior)

        DRPS[i] = DRPS_posterior 
        relDRPS[i] = DRPS_posterior/DRPS_prior  # relative DRPS
        #print(f"DRPS relative = {DRPS[i+Ndim]}")

    return tuple(DRPS), tuple(relDRPS)