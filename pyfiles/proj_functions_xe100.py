##### Phys 6041 Project - University of Cincinnati:  Fits for Dark Matter Direct Detection
##### Functions - hardcoded for xenon100 experiment

import numpy as np
import pyfiles.proj_constants as pc
import pyfiles.proj_functions as pf
from scipy.integrate import quad
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numba import njit, jit

v0 = pc.v0
mx = pc.mx
nx = pc.nx
nn = pc.xe_nn
ni = pc.xe_ni
mT = pc.xe_mT
uT = pc.xe_uT
nT = pc.xe_nT
Gi = pc.xe_Gi
t = pc.xe_t
nobs = pc.xe_nobs
nback = pc.xe_nback
err = pc.xe_err


## xsec from Mathematica

@jit
def xsec(ER):
    '''Cross-sections from Mathematica:  Xenon_100

    This produces the cross-sections obtained from Mathematica for given recoil energy.

    Parameters
    ----------
    ER : array (N)
         Recoil energy.

    Returns
    -------
    float[M,N]
        Array of cross section values for M isotopes.
    '''
    ER = np.asarray(ER)
    xsec = [(633091.1114941736 + ER*(-19487.244767546996 + ER*(237.321244490632 + ER*(-1.486803254822325 + ER*(0.00527808008700502 + ER*(-0.000010969603562960854 + ER*(1.3197446649654258e-8 + ER*(-8.604057524797386e-12 + ER*(2.5347354985179696e-15 + (-1.709919414433884e-19 + 3.308187086859928e-24*ER)*ER)))))))))/np.exp(0.01609826998366689*ER),
            (645067.7348483271 + ER*(-20176.020159666867 + ER*(248.08701026309387 + ER*(-1.5576502188802346 + ER*(0.005506739591658765 + ER*(-0.000011346760650527372 + ER*(1.352939816207266e-8 + ER*(-8.833328859021076e-12 + ER*(2.7387204605572095e-15 + (-2.712781055642179e-19 + 8.293204617796245e-24*ER)*ER)))))))))/np.exp(0.0162609589893489*ER),
            (657177.6611599155 + ER*(-20804.638417452552 + ER*(260.0894064350892 + ER*(-1.6693838667803618 + ER*(0.006064645378622305 + ER*(-0.00001290453912587759 + ER*(1.5954767516217733e-8 + ER*(-1.0815139215007741e-11 + ER*(3.4455231622632193e-15 + (-3.2449014017293167e-19 + 9.235465782489534e-24*ER)*ER)))))))))/np.exp(0.01642403195553838*ER),
            (669479.4884441359 + ER*(-21529.366343798072 + ER*(271.90221897845754 + ER*(-1.751583272533074 + ER*(0.006352245448377606 + ER*(-0.000013447332010273583 + ER*(1.6555556373503026e-8 + ER*(-1.1302481598551103e-11 + ER*(3.793103870191238e-15 + (-4.662998082056099e-19 + 1.8523713051087978e-23*ER)*ER)))))))))/np.exp(0.016587486924431398*ER),
            (681921.9210155475 + ER*(-22190.941340679394 + ER*(284.7822560787502 + ER*(-1.873814663437903 + ER*(0.006974238704620588 + ER*(-0.0000152159506899105 + ER*(1.9359524725313482e-8 + ER*(-1.3645562047855229e-11 + ER*(4.666540127206631e-15 + (-5.480441352528342e-19 + 2.0308335832480598e-23*ER)*ER)))))))))/np.exp(0.016751321963021332*ER),
            (707137.0912606611 + ER*(-23709.33686328353 + ER*(311.513851441322 + ER*(-2.0832613157333397 + ER*(0.007836801224395066 + ER*(-0.000017235811853646755 + ER*(2.2188737414878316e-8 + ER*(-1.61187563939412e-11 + ER*(6.033241928311872e-15 + (-9.586080994033606e-19 + 5.253267450184154e-23*ER)*ER)))))))))/np.exp(0.017080124638271407*ER),
            (733152.7069342799 + ER*(-25184.09967322127 + ER*(339.5411683873248 + ER*(-2.3348922094126134 + ER*(0.00905144878484389 + ER*(-0.000020562958198175655 + ER*(2.740771878907724e-8 + ER*(-2.0647031365664502e-11 + ER*(8.002155063063582e-15 + (-1.3029758941126175e-18 + 7.275983589154827e-23*ER)*ER)))))))))/np.exp(0.01741042499455707*ER)]
    xsec = np.asarray(xsec)
    return xsec


def xsec_plt(ER,**options):
    '''Plots the cross-sections from Mathematica:  Xenon_100

    This produces a plot of the cross-sections obtained from Mathematica for given recoil energy.

    Parameters
    ----------
    ER : array (N)
         Recoil energy.
    size: tuple (i,j), optional
          Size of the plot, width = i, height = j.
    cuts: array (i,j,k), optional
          Specifies which isotopes to plot, start = i, stop = j, interval = k.
          
    Returns
    -------
    matplotlib plot
        Plot of cross-sections for given isotopes.
    '''
    ER = np.asarray(ER)
    xsecf = np.asarray(xsec(ER))
    fig, axs = plt.subplots(1,figsize=options.get("size"))
    if options.get("cuts")==None:
        for i in range(len(nn)):
            axs.plot(ER,xsecf[i], label=f"$N_n$ = {nn[i]}")
            axs.legend()
    else:
        cuts = np.asarray(options.get("cuts"))
        cuts = np.arange(cuts[0],cuts[1],cuts[2])
        for i in cuts:
            axs.plot(ER,xsecf[i], label=f"$N_n$ = {nn[i]}")
            axs.legend()
    return 

@jit
def vlow(ER,mind):
    '''Minimum velocity for integration limits:  Xenon_100

    This produces the minimum velocity hardcoded for the xe100 experiment.

    Parameters
    ----------
    ER : array (N)
         Recoil energy.
    mind : int
           Index of mass desired from DM mass array.

    Returns
    -------
    float[N,M]
        Array of cross section values for M isotopes.
    '''
    ER = np.asarray(ER)
    vlow = []
    for i in range(len(ER)):
        vlow += [1/uT[mind]*np.sqrt(mT*ER[i]/2)]
    vlow = np.asarray(vlow)
    return vlow


def vlow_plt(ER,mind,**options):
    '''Plots the minimum velocity for integration limits:  Xenon_100

    This produces a plot of the minimum velocity hardcoded for the xe100 experiment.

    Parameters
    ----------
    ER : array (N)
         Recoil energy.
    mind : int
           Index of mass desired from DM mass array.
    size: tuple (i,j), optional
          Size of the plot, width = i, height = j.
    cuts_iso: array (i,j,k), optional
              Specifies which isotopes to plot, start = i, stop = j, interval = k.
          
    Returns
    -------
    matplotlib plot
        Plot of minimum velocities for given isotopes and dark matter masses.
    '''
    ER = np.asarray(ER)
    vlowf = np.asarray(vlow(ER,mind))
    fig, axs = plt.subplots(1,figsize=options.get("size"))
    if (options.get("cuts_iso")==None):
        for j in range(len(nn)):
            axs.plot(ER,vlowf[:,j], label=f"$N_n$ = {nn[j]}")
            axs.legend()
    else:
        cuts_iso = np.asarray(options.get("cuts_iso"))
        cuts_iso = np.arange(cuts_iso[0],cuts_iso[1],cuts_iso[2])
        for i in cuts_iso:
            axs.plot(ER,vlowf[:,i], label=f"$N_n$ = {nn[i]}")
            axs.legend()
    return 


@jit
def vintlMB(ER,mind):
    '''Velocity integral for xenon100 - Maxwell-Boltzmann

    This produces the velocity integral hardcoded for the xe100 experiment.

    Parameters
    ----------
    ER : array (N)
         Recoil energy.
    mind : int
           Index of mass desired from DM mass array.

    Returns
    -------
    float[N,M]
        Array of cross section values for M isotopes.
    '''
    ER = np.asarray(ER)
    vlowf = vlow(ER,mind)
    vint = []
    for i in range(len(ER)):
        for j in range(np.shape(vlowf)[1]):
            vint += [quad(pf.vintdvMB, vlowf[i,j], np.inf, args=(v0,))[0]]
    vint = np.reshape(np.asarray(vint),(len(ER),np.shape(vlowf)[1]))
    return vint


def vintlMB_plt(ER,mind,**options):
    '''Plots the velocity integral for Xenon_100

    This produces a plot of the velocity integral hardcoded for the xe100 experiment.

    Parameters
    ----------
    ER : array (N)
         Recoil energy.
    mind : int
           Index of mass desired from DM mass array.
    size: tuple (i,j), optional
          Size of the plot, width = i, height = j.
    cuts_iso: array (i,j,k), optional
              Specifies which isotopes to plot, start = i, stop = j, interval = k.
          
    Returns
    -------
    matplotlib plot
        Plot of velocity distribution integral for given isotopes and dark matter masses.
    '''
    ER = np.asarray(ER)
    vintf = np.asarray(vintlMB(ER,mind))
    fig, axs = plt.subplots(1,figsize=options.get("size"))
    if (options.get("cuts_iso")==None):
        for j in range(len(nn)):
            axs.plot(ER,vintf[:,j], label=f"$N_n$ = {nn[j]}")
            axs.legend()
    else:
        cuts_iso = np.asarray(options.get("cuts_iso"))
        cuts_iso = np.arange(cuts_iso[0],cuts_iso[1],cuts_iso[2])
        for i in cuts_iso:
            axs.plot(ER,vintf[:,i], label=f"$N_n$ = {nn[i]}")
            axs.legend()
    return 


## Rates
@jit
def drder(ER,mind):   
    '''Differntial rate per recoil energy for xenon100

    This produces the differential rate per recoil energy hardcoded for the xe100 experiment.

    Parameters
    ----------
    ER : array (N)
         Recoil energy.
    mind : int
           Index of mass desired from DM mass array.

    Returns
    -------
    float[N]
        Array of values.
    '''
    ER = np.asarray(ER)
    vint = np.asarray(vintlMB(ER,mind))
    xsecf = np.asarray(xsec(ER))
    drder = []
    for j in range(len(ER)):
        drder += [np.sum(nT*nx[mind]*vint[j,:]*ni*xsecf[:,j])]
    drder = np.asarray(drder)
    return drder


def drder_plt(ER,mind,**options):
    '''Plots the differential rate:  Xenon_100

    This produces a plot of the differential rate hardcoded for the xe100 experiment.

    Parameters
    ----------
    ER : array (N)
         Recoil energy.
    mind : int
           Index of mass desired from DM mass array.
    size: tuple (i,j), optional
          Size of the plot, width = i, height = j.
          
    Returns
    -------
    matplotlib plot
        Plot of differential rates for given DM masses.
    '''
    ER = np.asarray(ER)
    drderf = np.asarray(drder(ER,mind))
    fig, axs = plt.subplots(1,figsize=options.get("size"))
    axs.plot(ER,drderf)
    return 

@jit
def rate(ER,mind):
    '''Total rate from trapezoid integration

    This produces the total rate using the np.trapz method hardcoded for the xe100 experiment.

    Parameters
    ----------
    ER : array (N)
         Recoil energy values.
    mind : int
           Index of mass desired from DM mass array.

    Returns
    -------
    float[Q]
        Value of rate for Q energy bins.
    '''  
    ER = np.asarray(ER)
    drderf = np.asarray(drder(ER,mind))
    r = []
    for i in range(np.shape(Gi)[1]):
            r += [np.trapz(Gi[:,i]*drderf,ER)]
    r = np.asarray(r)
    return r



def chi2(param,ER,mind):
    '''Chi-squared function for number of events

    This produces a chi-squared function for the observed and theoretical number of events hardcoded for the xe100 experiment.

    Parameters
    ----------
    param : float
            Values from effective field theory.
    ER : array (N)
         Recoil energy values.
    mind : int
           Index of mass desired from DM mass array.

    Returns
    -------
    float
        Chi-squared value.
    '''
    ER = np.asarray(ER)
    def vlow(ER):
        vlow = []
        for j in range(len(mT)):
            for i in range(len(ER)):
                vlow += [1/uT[mind,j]*np.sqrt(mT[j]*ER[i]/2)]
        vlow = np.reshape(np.asarray(vlow),(len(ER),len(nn)))
        return vlow
    def vint(ER):
        vlowf = vlow(ER)
        vint = []
        for j in range(len(mT)):
            for i in range(len(ER)):
                vint += [quad(pf.vintdvMB, vlowf[i,j], np.inf, args=(v0,))[0]]
        vint = np.reshape(np.asarray(vint),(len(ER),len(nn)))
        return vint
    def drder(ER):
        vintf = vint(ER)
        xsecf = np.asarray(xsec(ER))
        drder = []
        for j in range(len(ER)):
            drder += [np.sum(nT*nx[mind]*vintf[j,:]*ni*xsecf[:,j])]
        drder = np.asarray(drder)
        return drder
    def rate(ER):
        drderf = drder(ER)
        r = []
        for i in range(np.shape(Gi)[1]):
            r += [np.trapz(Gi[:,i]*drderf,ER)]
        r = np.asarray(r)
        return r
    pts = len(nobs)
    ratef = np.asarray(rate(ER))
    npred = param**2*ratef*t
    chisq = np.sum((nobs-(nback + npred))**2/(err)**2) / (pts-1)
    chisq = np.asarray(chisq)
    return chisq


def chi2min(ER,mind,init,**options):
    '''Minimization of chi-squared function

    This performs a minimization on a given chi-squared function hardcoded for the xe100 experiment.

    Parameters
    ----------
    ER : array (N)
         Recoil energy values.
    mind : int
           Index of mass desired from DM mass array.
    init : array(S)
           Values for initial guesses of parameters.
         

    Returns
    -------
    res : OptimizeResult (scipy.optimize.OptimizeResult)
          The optimization result represented as a OptimizeResult object.
          
        x : ndarray
            The solution of the optimization.
        success : bool
                  Whether or not the optimizer exited successfully.
        status : int
                 Termination status of the optimizer. Its value depends on the underlying solver. Refer to message for details.
        message : str
                  Description of the cause of the termination.
        fun, jac, hess: ndarray
                        Values of objective function, its Jacobian and its Hessian (if available). The Hessians may be approximations, see the documentation of the function in question.
        hess_inv : object
                   Inverse of the objective function’s Hessian; may be an approximation. Not available for all solvers. The type of this attribute may be either np.ndarray or scipy.sparse.linalg.LinearOperator.
        nfev, njev, nhev : int
                           Number of evaluations of the objective functions and of its Jacobian and Hessian.
        nit : int
              Number of iterations performed by the optimizer.
        maxcv : float
                The maximum constraint violation.     
    ''' 
    ER = np.asarray(ER)
    def vlow(ER):
        vlow = []
        for j in range(len(mT)):
            for i in range(len(ER)):
                vlow += [1/uT[mind,j]*np.sqrt(mT[j]*ER[i]/2)]
        vlow = np.reshape(np.asarray(vlow),(len(ER),len(nn)))
        return vlow
    def vint(ER):
        vlowf = vlow(ER)
        vint = []
        for j in range(len(mT)):
            for i in range(len(ER)):
                vint += [quad(pf.vintdvMB, vlowf[i,j], np.inf, args=(v0,))[0]]
        vint = np.reshape(np.asarray(vint),(len(ER),len(nn)))
        return vint
    def drder(ER):
        vintf = vint(ER)
        xsecf = np.asarray(xsec(ER))
        drder = []
        for j in range(len(ER)):
            drder += [np.sum(nT*nx[mind]*vintf[j,:]*ni*xsecf[:,j])]
        drder = np.asarray(drder)
        return drder
    def rate(ER):
        drderf = drder(ER)
        r = []
        for i in range(np.shape(Gi)[1]):
            r += [np.trapz(Gi[:,i]*drderf,ER)]
        r = np.asarray(r)
        return r
    def chi2(param,ER):
        pts = len(nobs)
        ratef = rate(ER)
        npred = param**2*ratef*t
        chisq = np.sum((nobs-(nback + npred))**2/(err)**2) / (pts-1)
        return chisq
    bounds = options.get("bounds")
    method = options.get("method")
    minchi2 = minimize(chi2,init,args=(ER,),method=method,bounds=bounds)
    
    return minchi2