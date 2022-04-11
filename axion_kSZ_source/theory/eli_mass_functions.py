from .sigma_interpolation import SigmaInterpolator
from .sigma_interpolation_FFTLog import SigmaInterpolatorFFTLog
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

                :type sigmaInterpolator: SigmaInterpolatorFFTLog
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
        T = 1e4
        # mMin = (1.308695e-10) * ((self.cosmo.RHO_C*1000 * self.cosmo.OmegaM)**(-1/2)) * ((1+z)**(-3/2)) * (1e4)**(3/2) # Included Delta in pre-eval w/ cosmology.py OmegaM property and in solar masses
        mMin = (1e8/self.cosmo.h) * (10/(1+z) * T/(4e4))**(3/2)

        sigma_min = self.sigmaInt(mMin,z)
        sigma = self.sigmaInt(m,z)
        K = erfinv(1 - 1/zeta)
        B0 = self.cosmo.delta_crit - np.sqrt(2) * K * sigma_min
        B = B0 + K/(np.sqrt(2)*sigma_min) * sigma**2

        # print("K =",K)
        # print('sigma_min =',sigma_min)
        # print("C0 =",B0)
        # print("B0 =",B0*self.sigmaInt(m,0)/sigma)

        vals = np.sqrt(2 / np.pi) * (self.cosmo.rho_mean / m) * np.fabs(self.sigmaInt.dlogSigma_dlogm(m, z)) * (B0 / sigma) * np.exp(-B**2 / (2 * sigma**2))
        return vals

    def B(self, m, z, zeta=40):
        mMin = (1.308695e-10) * ((self.cosmo.RHO_C*1000 * self.cosmo.OmegaM)**(-1/2)) * ((1+z)**(3/2)) * (1e4)**(3/2) # Included Delta in pre-eval w/ cosmology.py OmegaM property and in solar masses
        sigma_min = self.sigmaInt(mMin,z)
        sigma = self.sigmaInt(m,z)
        K = erfinv(1-1/zeta)
        B0 = self.cosmo.delta_crit - np.sqrt(2) * K * sigma_min
        return B0 + K/(np.sqrt(2) * sigma_min) * sigma**2

    def B0(self, z, zeta=40):
        mMin = (1.308695e-10) * ((self.cosmo.RHO_C*1000 * self.cosmo.OmegaM)**(-1/2)) * ((1+z)**(3/2)) * (1e4)**(3/2) # Included Delta in pre-eval w/ cosmology.py OmegaM property and in solar masses
        sigma_min = self.sigmaInt(mMin,z)
        K = erfinv(1-1/zeta)
        return self.cosmo.delta_crit - np.sqrt(2) * K * sigma_min
