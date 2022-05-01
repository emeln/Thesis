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

cosmos = []
camb = []
lin_power = []
growth = []
sigmaInt = []
bubbleFunc = []
pressSchechter = []
bubbleFunc_2 = []
cosmo_names = []
cosmo_params = []

cosmo_dict = {
    'cosmos': cosmos,
    'camb': camb,
    'lin_power': lin_power,
    'growth': growth,
    'sigmaInt': sigmaInt,
    'bubbleFunc': bubbleFunc,
    'pressSchechter': pressSchechter,
    'bubbleFunc_2': bubbleFunc_2,
    'cosmo_names': cosmo_names,
    'cosmo_params': cosmo_params
}

z_vals_sigma = np.linspace(0,20,25)
window = 'sharp_k'
int_helper = IntegrationHelper(2048)

def update_dict(): # Update the cosmology dictionary & autosave
    global cosmo_dict,cosmos,camb,lin_power,growth,sigmaInt,bubbleFunc,pressSchechter,bubbleFunc_2

    cosmo_dict = {
        'cosmos': cosmos,
        'camb': camb,
        'lin_power': lin_power,
        'growth': growth,
        'sigmaInt': sigmaInt,
        'bubbleFunc': bubbleFunc,
        'pressSchechter': pressSchechter,
        'bubbleFunc_2': bubbleFunc_2,
        'cosmo_names': cosmo_names,
        'cosmo_params': cosmo_params
    }

    try:
        os.rename('autosave_1.pkl','autosave_2.pkl')
    except FileNotFoundError:
        pass
    try:
        os.rename('autosave_0.pkl','autosave_1.pkl')
    except FileNotFoundError:
        pass
    save_cosmology('autosave_0.pkl')

def save_cosmology(filename='cosmology.pkl'):
    # Pickle the current cosmologies for quicker access when starting a new kernel. Default file is 'cosmology.pkl', but a different one can be specified
    c_pkl = []
    for key in cosmo_dict:
        c_pkl.append(cosmo_dict[key])
    with open('newpickle.pkl','wb') as picklefile:
        try:
            pickle.dump(c_pkl,picklefile,pickle.HIGHEST_PROTOCOL)
        except pickle.PicklingError:
            print('Could not pickle! Probably cause of autoreload')
        else:
            os.rename('newpickle.pkl',filename)

def load_cosmology(filename='cosmology.pkl'):
    # Load a pickled cosmology into cosmo_dict, default is 'cosmology.pkl', but a different can be specified
    global cosmo_dict, cosmos, camb, lin_power, growth, sigmaIntFFT, bubbleFuncFFT, pressSchechterFFT, bubbleFuncFFT_2, cosmo_params, cosmo_names
    with open(filename,'rb') as picklefile:
        c_pkl = pickle.load(picklefile)
    for i, key in enumerate(cosmo_dict):
        cosmo_dict[key] = c_pkl[i]

    cosmos = cosmo_dict['cosmos']
    camb = cosmo_dict['camb']
    lin_power = cosmo_dict['lin_power']
    growth = cosmo_dict['growth']
    sigmaInt = cosmo_dict['sigmaInt']
    sigmaInt = cosmo_dict['bubbleFunc']
    pressSchechter = cosmo_dict['pressSchechter']
    bubbleFunc_2 = cosmo_dict['bubbleFunc_2']
    cosmo_names = cosmo_dict['cosmo_names']
    cosmo_params = cosmo_dict['cosmo_params']

def reset_cosmo():
    global cosmos,camb,lin_power,growth,sigmaInt,bubbleFunc,pressSchechter,bubbleFunc_2, cosmo_params, cosmo_names, cosmo_dict

    cosmos = []
    camb = []
    lin_power = []
    growth = []
    sigmaInt = []
    bubbleFunc = []
    pressSchechter = []
    bubbleFunc_2 = []
    cosmo_names = []
    cosmo_params = []

    cosmo_dict = {
        'cosmos': cosmos,
        'camb': camb,
        'lin_power': lin_power,
        'growth': growth,
        'sigmaInt': sigmaInt,
        'bubbleFunc': bubbleFunc,
        'pressSchechter': pressSchechter,
        'bubbleFunc_2': bubbleFunc_2,
        'cosmo_names': cosmo_names,
        'cosmo_params': cosmo_params
    }

def add_cosmo(cosmo,cosmo_name):
    global cosmos,camb,lin_power,growth,sigmaInt,bubbleFunc,pressSchechter,bubbleFunc_2, cosmo_params, cosmo_names
    cid = len(cosmos)
    outpath = "./sigma_tests/"
    fileroot ="test_"+cosmo_name
    log_path = outpath+"sigma_test_"+cosmo_name+"_log.log"

    thisCamb = AxionCAMBWrapper(outpath, fileroot, log_path)
    thisCamb(cosmo)
    camb.append(thisCamb)
    cosmo.set_H_interpolation(thisCamb)

    lin_power.append(thisCamb.get_linear_power(extrap_kmax=1e9, extrap_kmin=1e-6))
    growth.append(thisCamb.get_growth())

    thisSigmaInt = SigmaInterpolatorFFTLog(cosmo, lin_power[cid], growth[cid], z_vals_sigma, 1e-6, 1e9, Nr=1024, window_function=window)
    thisSigmaInt.compute()
    sigmaInt.append(thisSigmaInt)

    bubbleFunc.append(BubbleMassFunction(cosmo,thisSigmaInt))
    pressSchechter.append(EliMassFunction(cosmo,thisSigmaInt))

    bubbleFunc_2.append(BMF2(cosmo,thisSigmaInt))

    cosmos.append(cosmo)
    cosmo_names.append(cosmo_name)

    update_dict()

def update_cosmo(cosmo,i): # Updates the arrays' ith position with cosmo
    global cosmos,camb,lin_power,growth,sigmaInt,bubbleFunc,pressSchechter,bubbleFunc_2,cosmo_names
    thisCamb = camb[i]
    thisCamb(cosmo)
    camb[i] = thisCamb
    cosmo.set_H_interpolation(thisCamb)

    lin_power[i] = thisCamb.get_linear_power(extrap_kmax=1e9, extrap_kmin=1e-6)
    growth[i] = thisCamb.get_growth()

    thisSigmaInt =  SigmaInterpolatorFFTLog(cosmo, lin_power[i], growth[i], z_vals_sigma, 1e-6, 1e9, Nr=1024, window_function=window)
    thisSigmaInt.compute()
    sigmaInt[i] = thisSigmaInt

    bubbleFunc[i] = BubbleMassFunction(cosmo,thisSigmaInt)
    pressSchechter[i] = EliMassFunction(cosmo,thisSigmaInt)
    bubbleFunc_2[i] = BMF2(cosmo,thisSigmaInt)

    cosmos[i] = cosmo

    update_dict()

def set_window(window_function):
    global sigmaInt,pressSchechter,bubbleFunc,bubbleFunc_2,window
    window = window_function
    for cid, cosmo in enumerate(cosmos):
        sigmaInt[cid] = SigmaInterpolatorFFTLog(cosmo, lin_power[cid], growth[cid], z_vals_sigma, 1e-6, 1e9, Nr=1024, window_function=window)
        sigmaInt[cid].compute()
        pressSchechter[cid] = EliMassFunction(cosmo,sigmaInt[cid])
        bubbleFunc[cid] = BubbleMassFunction(cosmo,sigmaInt[cid])
        bubbleFunc_2[cid] = BMF2(cosmo,sigmaInt[cid])

def new_cosmo(name=None,m_ax=None, ax_frac=None, h_param=None, rhff=None):
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

    cid = len(cosmos)
    if name is None:
        name = f'cosmo{cid}'

    if name in cosmo_names:
        print(f'There is already a cosmology with name {name}')
        return

    this_cosmo = Cosmology.generate(m_axion=m,axion_frac=frac,h=h_param,read_H_from_file=read_h_from_file)
    add_cosmo(this_cosmo,name)

    new_params = {
        'm_ax': m_ax,
        'ax_frac': ax_frac,
        'h_param': h,
        'read_h_from_file': rhff
    }

    cosmo_params.append(new_params)

def update_cosmo(cosmo_name=None,cid=None,m_ax=None, ax_frac=None, h_param=None, rhff=None):
    if cosmo_name is None and cid is None:
        print('Must provide either a cosmology name or cid')
        return
    found_cosmo = False
    if cosmo_name is not None:
        if cid is not None:
            if cosmo_dict['cosmo_names'].index(cosmo_name) != cid:
                print('Cosmo name and cid do not match')
        else:
            cid = cosmo_dict['cosmo_names'].index(cosmo_name)
    if cid < len(cosmo_params):
        print(f'No cosmology with cid {cid}')
        return

    if axion_mass is not None:
        cosmo_params[cid]['m_ax'] = m_ax
    if axion_frac is not None:
        cosmo_params[cid]['ax_frac'] = ax_frac
    if h is not None:
        cosmo_params[cid]['h_param'] = h_param
    if read_h_from_file is not None:
        cosmo_params[cid]['read_h_from_file'] = rhff
    found_cid = True
