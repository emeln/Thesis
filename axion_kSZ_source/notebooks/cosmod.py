import sys
sys.path.append("../../")
import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import *

import dill as pickle

from axion_kSZ_source.axion_camb_wrappers.run_axion_camb import AxionCAMBWrapper
from axion_kSZ_source.theory.cosmology import Cosmology
from axion_kSZ_source.theory.sigma_interpolation_FFTLog import SigmaInterpolatorFFTLog
from axion_kSZ_source.auxiliary.integration_helper import IntegrationHelper
from axion_kSZ_source.auxiliary.window_functions import WindowFunctions
from axion_kSZ_source.theory.eli_mass_functions import EliMassFunction
from axion_kSZ_source.theory.eli_mass_functions import BubbleMassFunction
from axion_kSZ_source.theory.eli_mass_functions import BMF2

class CosmoDictionary:
    def __init__(self,dict=None):
        if dict is None:
            self.__cosmos = []
            self.__camb = []
            self.__lin_power = []
            self.__growth = []
            self.__sigmaInt = []
            self.__bubbleFunc = []
            self.__pressSchechter = []
            self.__bubbleFunc_2 = []
            self.__cosmo_names = []
            self.__cosmo_params = []
        else:
            self.__cosmos = dict['cosmos']
            self.__camb = dict['camb']
            self.__lin_power = dict['lin_power']
            self.__growth = dict['growth']
            self.__sigmaInt = dict['sigmaInt']
            self.__bubbleFunc = dict['bubbleFunc']
            self.__pressSchechter = dict['pressSchechter']
            self.__bubbleFunc_2 = dict['bubbleFunc_2']
            self.__cosmo_names = dict['cosmo_names']
            self.__cosmo_params = dict['cosmo_params']

        self.__cosmo_dict = {
            'cosmos': self.__cosmos,
            'camb': self.__camb,
            'lin_power': self.__lin_power,
            'growth': self.__growth,
            'sigmaInt': self.__sigmaInt,
            'bubbleFunc': self.__bubbleFunc,
            'pressSchechter': self.__pressSchechter,
            'bubbleFunc_2': self.__bubbleFunc_2,
            'cosmo_names': self.__cosmo_names,
            'cosmo_params': self.__cosmo_params
        }

        self.__z_vals_sigma = np.linspace(0,20,25)
        self.__window = 'sharp_k'
        self.__int_helper = IntegrationHelper(2048)
        self.__Nr = 1024

        self.__kmin = 1e-6
        self.__kmax = 1e9

    def __call__(self):
        return self.__cosmo_dict

    def update_dict(self): # Update the cosmology dictionary & autosave
        self.__cosmo_dict = {
            'cosmos': self.__cosmos,
            'camb': self.__camb,
            'lin_power': self.__lin_power,
            'growth': self.__growth,
            'sigmaInt': self.__sigmaInt,
            'bubbleFunc': self.__bubbleFunc,
            'pressSchechter': self.__pressSchechter,
            'bubbleFunc_2': self.__bubbleFunc_2,
            'cosmo_names': self.__cosmo_names,
            'cosmo_params': self.__cosmo_params
        }

        try:
            os.rename('autosave_1.pkl','autosave_2.pkl')
        except FileNotFoundError:
            pass
        try:
            os.rename('autosave_0.pkl','autosave_1.pkl')
        except FileNotFoundError:
            pass
        self.save_cosmology('autosave_0.pkl')

    def save_cosmology(self,filename='cosmology.pkl'):
        # Pickle the current cosmologies for quicker access when starting a new kernel. Default file is 'cosmology.pkl', but a different one can be specified
        c_pkl = []
        for key in self.__cosmo_dict:
            c_pkl.append(self.__cosmo_dict[key])

        c_pkl.append([self.__z_vals_sigma,self.__window,self.__int_helper,self.__kmin,self.__kmax,self.__Nr])

        with open('newpickle.pkl','wb') as picklefile:
            try:
                pickle.dump(c_pkl,picklefile,pickle.HIGHEST_PROTOCOL)
            except pickle.PicklingError:
                print('Could not pickle! Probably cause of autoreload')
            else:
                os.rename('newpickle.pkl',filename)

    def load_cosmology(self,filename='cosmology.pkl'):
        # Load a pickled cosmology into cosmo_dict, default is 'cosmology.pkl', but a different can be specified
        with open(filename,'rb') as picklefile:
            c_pkl = pickle.load(picklefile)
        for i, key in enumerate(self.__cosmo_dict):
            self.__cosmo_dict[key] = c_pkl[i]

        self.__cosmos = self.__cosmo_dict['cosmos']
        self.__camb = self.__cosmo_dict['camb']
        self.__lin_power = self.__cosmo_dict['lin_power']
        self.__growth = self.__cosmo_dict['growth']
        self.__sigmaInt = self.__cosmo_dict['sigmaInt']
        self.__bubbleFunc = self.__cosmo_dict['bubbleFunc']
        self.__pressSchechter = self.__cosmo_dict['pressSchechter']
        self.__bubbleFunc_2 = self.__cosmo_dict['bubbleFunc_2']
        self.__cosmo_names = self.__cosmo_dict['cosmo_names']
        self.__cosmo_params = self.__cosmo_dict['cosmo_params']

        self.__z_vals_sigma = c_pkl[-1][0]
        self.__window = c_pkl[-1][1]
        self.__int_helper = c_pkl[-1][2]
        self.__kmin = c_pkl[-1][3]
        self.__kmax = c_pkl[-1][4]
        self.__Nr = c_pkl[-1][5]

    def reset_cosmo(self):
        self.__cosmos = []
        self.__camb = []
        self.__lin_power = []
        self.__growth = []
        self.__sigmaInt = []
        self.__bubbleFunc = []
        self.__pressSchechter = []
        self.__bubbleFunc_2 = []
        self.__cosmo_names = []
        self.__cosmo_params = []

        self.__cosmo_dict = {
            'cosmos': self.__cosmos,
            'camb': self.__camb,
            'lin_power': self.__lin_power,
            'growth': self.__growth,
            'sigmaInt': self.__sigmaInt,
            'bubbleFunc': self.__bubbleFunc,
            'pressSchechter': self.__pressSchechter,
            'bubbleFunc_2': self.__bubbleFunc_2,
            'cosmo_names': self.__cosmo_names,
            'cosmo_params': self.__cosmo_params
        }

        self.__z_vals_sigma = np.linspace(0,20,25)
        self.__window = 'sharp_k'
        self.__int_helper = IntegrationHelper(2048)

        self.__kmin = 1e-6
        self.__kmax = 1e9

    def add_cosmo(self,cosmo,cosmo_name):
        cid = len(self.__cosmos)
        outpath = "./sigma_tests/"
        fileroot ="test_"+cosmo_name
        log_path = outpath+"sigma_test_"+cosmo_name+"_log.log"

        thisCamb = AxionCAMBWrapper(outpath, fileroot, log_path)
        thisCamb(cosmo)
        self.__camb.append(thisCamb)
        cosmo.set_H_interpolation(thisCamb.get_hubble())

        self.__lin_power.append(thisCamb.get_linear_power(extrap_kmax=self.__kmax, extrap_kmin=self.__kmin))
        self.__growth.append(thisCamb.get_growth())

        thisSigmaInt = SigmaInterpolatorFFTLog(cosmo, self.__lin_power[cid], self.__growth[cid], self.__z_vals_sigma, self.__kmin, self.__kmax, Nr=self.__Nr, window_function=self.__window)
        thisSigmaInt.compute()
        self.__sigmaInt.append(thisSigmaInt)

        self.__bubbleFunc.append(BubbleMassFunction(cosmo,thisSigmaInt))
        self.__pressSchechter.append(EliMassFunction(cosmo,thisSigmaInt))
        self.__bubbleFunc_2.append(BMF2(cosmo,thisSigmaInt))

        self.__cosmos.append(cosmo)
        self.__cosmo_names.append(cosmo_name)

        self.update_dict()

    def update_cosmo(self,cosmo_name=None,cid=None,m_ax=None, ax_frac=None, h_param=None, rhff=None):
        if cosmo_name is None and cid is None:
            print('Must provide either a cosmology name or cid')
            return

        if cosmo_name is not None:
            if cid is not None:
                if self.__cosmo_dict['cosmo_names'].index(cosmo_name) != cid:
                    print('Cosmo name and cid do not match')
            else:
                cid = self.__cosmo_dict['cosmo_names'].index(cosmo_name)
        if cid < len(self.__cosmo_params):
            print(f'No cosmology with cid {cid}')
            return

        if axion_mass is not None:
            self.__cosmo_params[cid]['m_ax'] = m_ax
        if axion_frac is not None:
            self.__cosmo_params[cid]['ax_frac'] = ax_frac
        if h is not None:
            self.__cosmo_params[cid]['h_param'] = h_param
        if read_h_from_file is not None:
            self.__cosmo_params[cid]['read_h_from_file'] = rhff

        cosmo = Cosmology.generate(m_axion=m_ax,axion_frac=ax_frac,h=h_param,read_H_from_file=rhff)

        thisCamb = self.__camb[cid]
        thisCamb(cosmo)
        self.__camb[cid] = thisCamb
        cosmo.set_H_interpolation(thisCamb.get_hubble())

        self.__lin_power[cid] = thisCamb.get_linear_power(extrap_kmax=self.__kmax, extrap_kmin=self.__kmin)
        self.__growth[cid] = thisCamb.get_growth()

        thisSigmaInt =  SigmaInterpolatorFFTLog(cosmo, self.__lin_power[cid], self.__growth[cid], self.__z_vals_sigma, self.__kmin, self.__kmax, Nr=self.__Nr, window_function=self.__window)
        thisSigmaInt.compute()
        self.__sigmaInt[cid] = thisSigmaInt

        self.__bubbleFunc[cid] = BubbleMassFunction(cosmo,thisSigmaInt)
        self.__pressSchechter[cid] = EliMassFunction(cosmo,thisSigmaInt)
        self.__self.__bubbleFunc_2[cid] = BMF2(cosmo,thisSigmaInt)

        self.__cosmos[cid] = cosmo

        self.update_dict()

    def new_cosmo(self,name=None,m_ax=None, ax_frac=None, h_param=None, rhff=None):
        if m_ax is None:
            m = 1e-24
        else:
            m = m_ax
        if ax_frac is None:
            frac = 5e-8
        else:
            frac = ax_frac
        if h_param is None:
            h = 0.72
        else:
            h = h_param
        if rhff is None:
            read_h_from_file = True
        else:
            read_h_from_file = rhff

        cid = len(self.__cosmos)
        if name is None:
            name = f'cosmo{cid}'

        if name in self.__cosmo_names:
            print(f'There is already a cosmology with name {name}')
            return

        this_cosmo = Cosmology.generate(m_axion=m,axion_frac=frac,h=h_param,read_H_from_file=read_h_from_file)
        self.add_cosmo(this_cosmo,name)

        new_params = {
            'm_ax': m_ax,
            'ax_frac': ax_frac,
            'h_param': h,
            'read_h_from_file': rhff
        }

        self.__cosmo_params.append(new_params)

    def update_k_range(self,kmin,kmax):
        if kmax < kmin:
            print('Minimum k is first parameter, maximum k is second')
            return

        for cid, cosmo in enumerate(self.__cosmos):
            self.__lin_power[cid] = self.__camb[cid].get_linear_power(extrap_kmax=kmax,extrap_kmin=kmin)
            thisSigmaInt = SigmaInterpolatorFFTLog(cosmo, self.__lin_power[cid], self.__growth[cid], self.__z_vals_sigma, kmin, kmax, Nr=1024, window_function=self.__window)
            thisSigmaInt.compute()
            self.__sigmaInt[cid] = thisSigmaInt

            self.__bubbleFunc[cid] = BubbleMassFunction(cosmo,thisSigmaInt)
            self.__pressSchechter[cid] = EliMassFunction(cosmo,thisSigmaInt)
            self.__bubbleFunc_2[cid] = BMF2(cosmo,thisSigmaInt)
            self.__kmin = kmin
            self.__kmax = kmax

        self.update_dict()

    def update_Nr(self,Nr):
        for cid, cosmo in enumerate(self.__cosmos):
            thisSigmaInt = SigmaInterpolatorFFTLog(cosmo, self.__lin_power[cid], self.__growth[cid], self.__z_vals_sigma, self.__kmin, self.__kmax, Nr=Nr, window_function=self.__window)
            thisSigmaInt.compute()
            self.__sigmaInt[cid] = thisSigmaInt

            self.__bubbleFunc[cid] = BubbleMassFunction(cosmo,thisSigmaInt)
            self.__pressSchechter[cid] = EliMassFunction(cosmo,thisSigmaInt)
            self.__bubbleFunc_2[cid] = BMF2(cosmo,thisSigmaInt)

            self.__Nr = Nr

        self.update_dict()

    def change_name(self, i, name):
        self.__cosmo_names[i] = name
        self.update_dict()

    def integrate(self, func, xmin, xmax):
        return self.__int_helper.integrate(func,xmin,xmax)

    @property
    def cosmos(self):
        return self.__cosmos

    @property
    def camb(self):
        return self.__camb

    @property
    def lin_power(self):
        return self.__lin_power

    @property
    def growth(self):
        return self.__growth

    @property
    def sigmaInt(self):
        return self.__sigmaInt

    @property
    def bubbleFunc(self):
        return self.__bubbleFunc

    @property
    def pressSchechter(self):
        return self.__pressSchechter

    @property
    def bubbleFunc_2(self):
        return self.__bubbleFunc_2

    @property
    def cosmo_names(self):
        return self.__cosmo_names

    @property
    def cosmo_params(self):
        return self.__cosmo_params

    @property
    def k_range(self):
        return [self.__kmin,self.__kmax]

    @property
    def k_min(self):
        return self.__kmin

    @property
    def k_max(self):
        return self.__kmax

    @property
    def z_vals_sigma(self):
        return self.__z_vals_sigma

    @property
    def window(self):
        return self.__window

    @property
    def int_helper(self):
        return self.__int_helper

    @k_min.setter
    def k_min(self,k):
        self.__k_min = k

    @k_max.setter
    def k_max(self,k):
        self.__k_max = k

    @z_vals_sigma.setter
    def z_vals_sigma(self,new_z):
        self.__z_vals_sigma = new_z
        for i, cosmo in enumerate(self.__cosmos):
            thisSigmaInt = SigmaInterpolatorFFTLog(cosmo, self.__lin_power[i], self.__growth[i], self.__z_vals_sigma, self.__kmin, self.__kmax, Nr=1024, window_function=self.__window)
            thisSigmaInt.compute()
            self.__sigmaInt[i] = thisSigmaInt
            self.__bubbleFunc[i] = BubbleMassFunction(cosmo, self.__sigmaInt[i])
            self.__pressSchechter[i] = EliMassFunction(cosmo, self.__sigmaInt[i])
            self.__bubbleFunc_2[i] = BMF2(cosmo, self.__sigmaInt[i])

        self.update_dict()

    @window.setter
    def window(self,new_window):
        self.__window = new_window
        for i, cosmo in enumerate(self.__cosmos):
            self.__sigmaInt[i] = SigmaInterpolatorFFTLog(cosmo, self.__lin_power[i], self.__growth[i], self.__z_vals_sigma, 1e-6, 1e9, Nr=1024, window_function=self.__window)
            self.__sigmaInt[i].compute()
            self.__pressSchechter[i] = EliMassFunction(cosmo,self.__sigmaInt[i])
            self.__bubbleFunc[i] = BubbleMassFunction(cosmo,self.__sigmaInt[i])
            self.__bubbleFunc_2[i] = BMF2(cosmo,self.__sigmaInt[i])

        self.update_dict()
