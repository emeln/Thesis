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
        # Output is m dn/dm = dn/dlogm
        mMin = (1.308695e-10) * ((self.cosmo.RHO_C*1000 * self.cosmo.OmegaM)**(-1/2)) * ((1+z)**(-3/2)) * (1e4)**(3/2)

        sigma_min = self.sigmaInt(mMin,z)
        sigma = self.sigmaInt(m,z)

        vals = np.sqrt(2 / np.pi) * (self.cosmo.rho_mean / m) * np.fabs(self.sigmaInt.dlogSigma_dlogm(m, z)) * (self.B0(m,z,zeta) / sigma) * np.exp(-self.B(m,z,zeta)**2 / (2 * sigma**2))
        return vals

    def B(self, m, z, zeta=40):
        return self.B0(m, z, zeta) + self.B1(m, z, zeta)*self.sigmaInt(m,z)**2

    def B0(self, m, z, zeta=40):
        K = erfinv(1 - 1/zeta)
        return self.cosmo.delta_crit - np.sqrt(2) * K * self.sigma_min(z)

    def B1(self, m, z, zeta=40):
        K = erfinv(1 - 1/zeta)
        return K/(np.sqrt(2)*self.sigma_min(z)) * self.sigmaInt(m,z)**2


    def sigma_min(self,z):
        mMin = (1.308695e-10) * ((self.cosmo.RHO_C*1000 * self.cosmo.OmegaM)**(-1/2)) * ((1+z)**(-3/2)) * (1e4)**(3/2)
        return self.sigmaInt(mMin,z)

class BMF2(MassFunction):
    def __call__(self, m, z, zeta=40):
        # Output is m dn/dm = dn/dlogm
        sigma = self.sigmaInt(m,0)

        vals = np.sqrt(2 / np.pi) * (self.cosmo.rho_mean / m) * np.fabs(self.sigmaInt.dlogSigma_dlogm(m, z)) * (self.B0(self, m, z, zeta) / sigma) * np.exp(-self.B(m,z,zeta)**2 / (2 * sigma**2))
        return vals

    def B(self, m, z, zeta=40):
        return self.B0(m,z,zeta) + self.B1(m,z,zeta)*self.sigmaInt(m,0)**2

    def B0(self, m, z, zeta=40):
        delta_c = self.cosmo.delta_crit * self.sigmaInt(m,0) / self.sigmaInt(m,z)
        K = erfinv(1 - 1/zeta)
        return delta_c - np.sqrt(2) * K * self.sigma_min(0)

    def B1(self, m, z, zeta=40):
        K = erfinv(1 - 1/zeta)
        return K/(np.sqrt(2)*sigma_min(0)) * self.sigmaInt(m,0)**2

    def sigma_min(self,z):
        mMin = (1.308695e-10) * ((self.cosmo.RHO_C*1000 * self.cosmo.OmegaM)**(-1/2)) * ((1+z)**(-3/2)) * (1e4)**(3/2)
        return self.sigmaInt(mMin,z)
