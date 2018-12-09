##### Phys 6041 Project - University of Cincinnati:  Fits for Dark Matter Direct Detection
##### General functions for various experiments

##### import necessary modules
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize


##### Velocity functions
#### velocity integral
def vintdvMB(v,v0):
    '''Define velocity distribution to integrate:  distribution times velocity

    This produces the velocity distribution multiplied by velocity, to be integrated over energy to get rates.

    Parameters
    ----------
    v : array (N)
        Velocity at which to evaluate distribution.
    v0 : float
         Value for Maxwell-Boltzmann velocity distribution.

    Returns
    -------
    float[N]
        Array of values evaluated at v.
    '''
    v = np.asarray(v)
    vint = np.sqrt(2/np.pi)*np.exp(-v**2/(2*v0**2))*v / (2*v0**3)
    return vint


def vlow(ER, mT, uT):
    '''Minimum velocity

    This produces the minimum velocity for the velocity integral given a recoil energy, target mass, and target-DM reduced mass.

    Parameters
    ----------
    ER : array (N)
         Recoil energy values.
    mT : float
         Target mass.
    uT : float
         Target-DM reduced mass.

    Returns
    -------
    array[N]
        Array of minimum velocity values.
    '''
    ER = np.asarray(ER)
    vlow = 1/uT*np.sqrt(mT*ER/2)
    return vlow

def vintlMB(ER, mT, uT, v0):
    '''Integral of velocity distribution:  Maxwell-Boltzmann

    This produces the integral of the velocity distribution used, limits dependent on recoil energy.

    Parameters
    ----------
    ER : array (N)
         Recoil energy values.
    mT : float
         Target mass.
    uT : float
         Target-DM reduced mass.
    v0 : float
         Choice for MB velocity distribution standard deviation.

    Returns
    -------
    float[N]
        Array of velocity integral values.
    '''
    ER = np.asarray(ER)
    vlowf = vlow(ER, mT, uT)
    vint = []
    for i in range(len(ER)):
        vint += [quad(vintdvMB, vlowf[i], np.inf, args=(v0,))[0]]
    vint = np.asarray(vint)
    return vint


## Rates
def drder(ER,mT,uT,v0,xsec,nT,nx,ni):   
    '''Differential rate per recoil energy

    This produces the differential rate per recoil energy for given velocity integrals and cross-sections

    Parameters
    ----------
    ER : array (N)
         Recoil energy values.
    mT : float
         Target mass.
    uT : float
         Target-DM reduced mass.
    v0 : float
         Choice for MB velocity distribution standard deviation.
    xsec : array (N)
           Cross-section values - should depend on recoil energy.
    nT : float
         Number of target particles.
    nx : float
         Local DM number density.
    ni : float
         Relative abundance of isotope.

    Returns
    -------
    float[N]
        Array of values for given recoil energies.
    '''      
    ER = np.asarray(ER)
    xsec = np.asarray(xsec)
    vint = vintlMB(ER, mT, uT, v0)
    drder = nT*nx*vint*ni*xsec
    return drder

def rate(ER,mT,uT,v0,xsec,nT,nx,ni,Gi):
    '''Total rate from trapezoid integration

    This produces the total rate using the np.trapz method

    Parameters
    ----------
    ER : array (N)
         Recoil energy values.
    mT : float
         Target mass.
    uT : float
         Target-DM reduced mass.
    v0 : float
         Choice for MB velocity distribution standard deviation.
    xsec : array (N)
           Cross-section values - should depend on recoil energy.
    nT : float
         Number of target particles.
    nx : float
         Local DM number density.
    ni : float
         Relative abundance of isotope.
    Gi : array(N,M)
         Detector efficiency for given recoil energy.

    Returns
    -------
    float[M]
        Value of rate for M energy bins.
    '''  
    ER = np.asarray(ER)
    drderf = drder(ER,mT,uT,v0,xsec,nT,nx,ni)
    Gi = np.asarray(Gi)
    r = []
    for i in range(np.shape(Gi)[1]):
        r += [np.trapz(Gi[:,i]*drderf,ER)]
    r = np.asarray(r)
    return r


## Chi-sq and fit
def chi2(param,ER,mT,uT,v0,xsec,nT,nx,ni,Gi,nobs,nback,err,t):
    '''Chi-squared function for number of events

    This produces a chi-squared function for the observed and theoretical number of events

    Parameters
    ----------
    param : array(P)
            Values from effective field theory.
    ER : array (N)
         Recoil energy values.
    mT : float
         Target mass.
    uT : float
         Target-DM reduced mass.
    v0 : float
         Choice for MB velocity distribution standard deviation.
    xsec : array (N)
           Cross-section values - should depend on recoil energy.
    nT : float
         Number of target particles.
    nx : float
         Local DM number density.
    ni : float
         Relative abundance of isotope.
    Gi : array(N,M)
         Detector efficiency for given recoil energy.
    nobs : array(N)
           Number of observed events for given energy bin.
    nback : array(N)
            Number of background events for given energy bin.
    err : array(N)
          Error in observed events for given energy bin.
    t : float
        Exposure time for given experiment.

    Returns
    -------
    float[P]
        Chi-squared value for P parameters.
    '''
    pts = len(nobs)
    ER = np.asarray(ER)
    param = np.asarray(param)
    ratef = rate(ER,mT,uT,v0,xsec,nT,nx,ni,Gi)
    ratef = np.asarray(ratef)
    nobs = np.asarray(nobs)
    nback = np.asarray(nback)
    err = np.asarray(err)
    npred = []
    for i in range(len(param)):
        npred += [param[i]**2*ratef*t]
    npred = np.asarray(npred)
    chisq = []
    for i in range(len(param)):
        chisq += [np.sum((nobs-(nback + npred[i,:]))**2/(err)**2) / (pts-1)]
    chisq = np.asarray(chisq)
    return chisq

def chi2min(init,ER,mT,uT,v0,xsec,nT,nx,ni,Gi,nobs,nback,err,t):
    '''Minimization of chi-squared function

    This performs a minimization on a given chi-squared function.

    Parameters
    ----------
    init : array(N)
           Values for initial guesses of parameters.
    ER : array (N)
         Recoil energy values.
    mT : float
         Target mass.
    uT : float
         Target-DM reduced mass.
    v0 : float
         Choice for MB velocity distribution standard deviation.
    xsec : array (N)
           Cross-section values - should depend on recoil energy.
    nT : float
         Number of target particles.
    nx : float
         Local DM number density.
    ni : float
         Relative abundance of isotope.
    Gi : array(N,M)
         Detector efficiency for given recoil energy.
    nobs : array(N)
           Number of observed events for given energy bin.
    nback : array(N)
            Number of background events for given energy bin.
    err : array(N)
          Error in observed events for given energy bin.
    t : float
        Exposure time for given experiment.

    Returns
    -------
    res : OptimizeResult
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
    Gi = np.asarray(Gi)
    nobs = np.asarray(nobs)
    nback = np.asarray(nback)
    err = np.asarray(err)
    def chisq(param,ER,mT,uT,v0,xsec,nT,nx,ni,Gi,nobs,nback,err,t):
        chisq = chi2(param,ER,mT,uT,v0,xsec,nT,nx,ni,Gi,nobs,nback,err,t)
        return chisq
    minchi2 = minimize(chisq, init, args=(ER,mT,uT,v0,xsec,nT,nx,ni,Gi,nobs,nback,err,t), method='Nelder-mead')
    return minchi2