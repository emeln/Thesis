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
        # Output is dN/dlogm
        vals = np.sqrt(2 / np.pi) * (self.cosmo.rho_mean * self.cosmo.delta_crit / self.sigmaInt(m,z) / m) * np.fabs(self.sigmaInt.dlogSigma_dlogm(m, z)) * np.exp(-self.cosmo.delta_crit**2 / (2 * self.sigmaInt(m, z)**2))
        return vals

class BubbleMassFunction(MassFunction):
    def __call__(self, m, z, zeta=40):
        # Output is m dn/dm

        m_p = 1.672619e-27  #  kg
        G = 6.67408e-11 #  m^3 kg^-1 s^-2
        kb = 1.380649e-23 #  J K^-1

        mMin = 1.38e6 * (kb / (G * m_p))**(3/2) * (overdensity(z))**(-1/2)
        B = self.cosmo.delta_crit - np.sqrt(2) * erfinv(1 - 1/zeta) * ()
        vals = np.sqrt(2 / np.pi) * (self.cosmo.rho_mean / m) * np.fabs(dlosSigma_dlogm(m, z)) * (B0 / self.sigmaInt(m, z)) * np.exp(-B**2 / (2 * (self.sigmaInt(m, z))**2))
