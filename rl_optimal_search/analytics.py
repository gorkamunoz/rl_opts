# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/analytics.ipynb.

# %% auto 0
__all__ = ['pdf_multimode', 'pdf_powerlaw', 'pdf_discrete_sample', 'get_policy']

# %% ../nbs/analytics.ipynb 1
import numpy as np

# %% ../nbs/analytics.ipynb 4
def pdf_multimode(L_max: int, # Maximum L for which probability is calculated
                     num_modes: int, # Number of modes 
                     lambdas: list, # Scales of each modes
                     probs: list # Probability weight of each mode
                    )-> np.array: # Array with probability of each L
    '''  Computes the discrete PDF of multi-mode exponential of the form
    
    SUM_i[ probs_i (1-Exp(-1/lambda_i)) Exp(-(L-1)/lambda_i) ]
    
    '''
    pdf = [np.sum((probs)*(np.exp(1/lambdas)-1)*np.exp(-L/lambdas)) for L in np.arange(1, L_max)]
    return pdf/np.sum(pdf)

# %% ../nbs/analytics.ipynb 7
from scipy.special import zeta

def pdf_powerlaw(L_max: int, # Maximum L for which probability is calculated
                 alpha: float = 1, # Exponent of the power law
                )-> np.array : # Array with probability of each L
    ''' Computes the discrete PDF of a powerlaw of the form 
    P(L) = L^(-alpha-1)'''
    return (1/zeta(alpha+1, q = 1))*np.arange(1,L_max).astype(float)**(-1-alpha)

# %% ../nbs/analytics.ipynb 9
def pdf_discrete_sample(pdf_func: object, # Function generating the pdf
                         num_samples: int, # Number of samples to create
                         **args_func # Arguments of the generating funcion
                        )-> np.array: # Samples
    ''' Samples discrete values from a given PDF'''
    P_L = pdf_func(**args_func)
    
    return np.random.choice(np.arange(1, len(P_L)+1), p = P_L, size = num_samples)
    

# %% ../nbs/analytics.ipynb 13
from scipy.special import zeta

def get_policy(n_max, # Maximum counter n_max for which the policy is calculated
               func, # Function generating the pdf
               ps_0: int = 1, # Value of the policy at L = 0 (should be one)
               **args_func # Arguments of the generating funcion (should have L_max as input parameter)
              )-> np.array : # Policy at each counter value
    ''' Given a PDF of step lengths, calculates the corresponding policy'''
    
    ps = np.zeros(n_max)
    ps[0] = ps_0  
    
    prob_L = func(L_max = n_max+1, **args_func)
    
    for l, p_lm1 in zip(range(2, L+1), prob_L):
        # l starts at 2 but prob_L starts at 1, because we want to divice by P(l-1)
                
        # Product
        prod = np.prod(ps[:l-1])
        # all together
        ps[l-1] = 1-p_lm1/prod
        
    return ps
