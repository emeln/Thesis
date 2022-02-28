from .sigma_interpolation import SigmaInterpolator
from .cosmology import Cosmology
from scipy.special import erfinv

# Notes:
# - Talked about this before, but Furlanetto has delta as a function of z and
#   sigma not as a function of z, whereas here delta is not a funciton of z but
#   sigma is a function of z
# - Check if mMin is done well
# - Need to figure out the overdensity term in mMin via Virial Thm. Need delta
#   at which

import numpy as np

class MassFunction(object):
    def __init__(self, cosmo, sigmaInterpolator):
        """

                :type sigmaInterpolator: SigmaInterpolator
                :type cosmo: Cosmology
                """
        self.cosmo = cosmo
        self.sigmaInt = sigmaInterpolator

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This is not implemented here! Use sublcasses!")

class PressSchechterMassFunction(MassFunction):
    def __call__(self, m, z):
        #output is dN/d log m
        vals = np.sqrt(2 / np.pi) * (self.cosmo.rho_mean * self.cosmo.delta_crit / self.sigmaInt(m, z) / m) * np.fabs(self.sigmaInt.dlogSigma_dlogm(m, z)) * np.exp(-self.cosmo.delta_crit**2 / (2 * self.sigmaInt(m, z)**2))
        return vals

class JenkinsMassFunction(MassFunction):
    def __call__(self, m, z):
        f = 0.315 * np.exp(-np.fabs(np.log(1 / self.sigmaInt(m, z)) + 0.61)**3.8)
        vals = self.cosmo.rho_mean / m * f * np.fabs(self.sigmaInt.dlogSigma_dlogm(m, z))
        return vals

class EliMassFunction(MassFunction):
    def __call__(self, m, z):
        #output is dN/d log m
        vals = np.sqrt(2 / np.pi) * (self.cosmo.rho_mean * self.cosmo.delta_crit / self.sigmaInt(m, z) / m) * np.fabs(self.sigmaInt.dlogSigma_dlogm(m, z)) * np.exp(-self.cosmo.delta_crit**2 / (2 * self.sigmaInt(m, z)**2))
        return vals

class BubbleMassFunction(MassFunction):
    def __call__(self, m, z, zeta=40):
        # Output is m dn/dm
        m_p = 1.672619e-27  #  kg
        G = 6.67408e-11 #  m^3 kg^-1 s^-2
        kb = 1.380649e-23 #  J K^-1
        delta = 178
        Omega_m = 0.315 # Not sure if this is the right one from Planck. This is in the abstract, but they have other Omega_m values
        # cosmo.RHO_C is the number given times h^2; should I account for that h^2 or leave it there?
        # I don't see an Omega_m in cosmology.py, so I've made my own, sourced from Planck 2018.
        # Actually there is the OmegaM property in cosmo, so could use that, although idk what sort of value it returns yet.

        T = 1e4 # virial temp in K that Furlanetto associates with minimum mass of ionizing source
        # mMin = (9/(2*np.sqrt(pi))) * ((kb/(G*m_p))**(3/2)) * ((delta * self.cosmo.RHO_C*1000 * Omega_m * (1/self.cosmo.h)**2)**(-1/2)) * ((1+z)**(-3/2)) * (T**(3/2)) # Everything computed here using Omega_m
        # mMin = (9/(2*np.sqrt(pi))) * ((kb/(G*m_p))**(3/2)) * ((delta * self.cosmo.RHO_C*1000 * self.cosmo.OmegaM)**(-1/2)) * ((1+z)**(-3/2)) * (T**(3/2)) # Everything computed here using Omega_m
        # mMin = (3.49203e21) * ((delta * self.cosmo.RHO_C*1000 * Omega_m * (1/self.cosmo.h)**2)**(-1/2)) * ((1+z)**(-3/2)) * (T**(3/2)) # Pre-evaluated (9/(2*np.sqrt(pi))) * ((kb/(G*m_p))**(3/2)) in Mathematica
        # mMin = (3.49203e21) * ((delta * self.cosmo.RHO_C*1000 * self.cosmo.OmegaM)**(-1/2)) * ((1+z)**(-3/2)) * T**(3/2) # Pre-evaluated w/ cosmology.py OmegaM property
        mMin = (2.61739e20) * ((self.cosmo.RHO_C*1000 * self.cosmo.OmegaM)**(-1/2)) * ((1+z)**(-3/2)) * T**(3/2) # Included Delta in pre-eval w/ cosmology.py OmegaM property
        # mMin = 1.91916e33 * (self.cosmo.OmegaM)**(-1/2) * ((1+z)**(-3/2)) * T**(3/2) # Pre-evaluate everything but cosmology.py OmegaM property
        # mMin = 1.91916e33 * Omega_m**(-1/2) * ((1+z)**(-3/2)) * T**(3/2) # Pre-evaluate everything but Omega_m

        sigma = self.sigmaInt(m,z)
        sigma_min = self.sigmaInt(mMin,z)
        K = erfinv(1 - 1/zeta)
        B0 = self.cosmo.delta_crit - np.sqrt(2) * K * sigma_min
        B = B0 + K/(np.sqrt(2) * sigma_min)
        vals = np.sqrt(2 / np.pi) * (self.cosmo.rho_mean / m) * np.fabs(self.sigmaInt.dlogSigma_dlogm(m, z)) * (B0 / sigma) * np.exp(-B**2 / (2 * sigma**2))
        return vals
