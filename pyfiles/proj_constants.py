##### Phys 6041 Project - University of Cincinnati:  Fits for Dark Matter Direct Detection
##### Constants

##### import necessary modules
import numpy as np


v0 = np.float64(236.0*10**3/(3*10**8))   ## MB - c
mx = np.array([10,50,100,500,1000])   ## mass of DM - GeV
nx = 2.3059*10**-42/mx       ## local DM number density - GeV^3

####Standard constants dictionary
stand_const = {'Maxwell-Boltzmann velocity - v0': v0}
stand_const_arrays = {'DM mass - mx': mx, 'DM number density - nx': nx}



### xenon100 isotopes
## number of nucleons 
xe_nn = np.array([128,129,130,131,132,134,136])   
## natural isotope abundances 
xe_ni = np.array([0.0191,0.26401,0.04071,0.21232,
               0.26909,0.10436,0.08857])


## mass of nuclei:  mT_i=0.938GeV*nn_i
xe_mT = xe_nn*0.938                  


## reduced nuclei-DM mass: uT_i=mT_i*mx_i/(mT_i+mx_i)
xe_uT = []
for i in range(0,len(mx)):
    xe_uT += [mx[i]*xe_mT/(mx[i]+xe_mT)]
xe_uT = np.asarray(xe_uT)  


## number of target particles:
## nT_i=34kg*(3*10^8m/s)^2/1.6*10^-10C/(mT_i*ni_i)
xe_nT = 34 * (3*10**8)**2 / (1.60217662*10**-10) / xe_mT * xe_ni                                      

## xenon dictionary
xe_const_arrays = {'nucleons - xe_nn': xe_nn, 'isotope abundance - xe_ni': xe_ni, 'mass - xe_mT': xe_mT, 'reduced mass - xe_uT': xe_uT, 'targets - xe_nT': xe_nT}



## detector response table                           
xe_data = np.loadtxt("xenon100_detector_table.dat")
## energies
xe_En = 10**-6*xe_data[:,0] 
## efficiencies
xe_Gi = xe_data[:,1:]   

## exposure time: t=live_days*secs/days/(GeV*secs)                         
xe_t = np.float64(224.6*24*3600/(6.58*10**-25))
    

##### define data to be compared w/ prediction
## number of observed events
xe_nobs = np.array([20,17,11,1,1,0,0,0,0])   
## number of background events           
xe_nback = np.array([24,16,12,1.1,1.0*10**-1,0.8*10**-1,
                     0.9,3.5*10**-1,1.8*10**-1])   
## error in observed events           
xe_err = np.array([5,3,3,0.3,0.5*10**-1,0.4*10**-1,0.3,
                   1.2*10**-1,0.7*10**-1])

## xenon100 dictionary
xe_100_const = {'exposure - xe_t': xe_t}
xe_100_const_arrays = {'energy - xe_En': xe_En, 'efficiency - xe_Gi': xe_Gi, 'observed events - xe_nobs': xe_nobs, 'background events - xe_nback': xe_nback, 'error - xe_err': xe_err}