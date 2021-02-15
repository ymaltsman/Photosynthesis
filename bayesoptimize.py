import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
#bayesian optimization of kinetic parameters

#Parameter values

#from Farquhar 1980
#turnover rate for carboxylation
kc = 2.5
#turnover rate for oxygenation
ko = .21*kc
#light intensity
I = 150
#total enzyme concentration
Et = 87.2
#CO2 concentration
C = 230
#O2 concentration
O = 210

#from Ye 2020
alpha = .295
beta = 2.42*(10**(-3))
gamma = 1.26*(10**(-4))

#couldnt find turnover # for RuBP so am setting kr, k2 = kc
kprod = kc
kE = kc

#also guesses here
kERC = .9*kc
kERO = .9*ko


#Coding the revised model
def dRdtalt(R, alphaa, alphab, kr =kc,  C = 230, O =210, kc = kc, ko = ko, I = I, Et = Et, alpha = alpha, beta = beta, gamma = gamma, kE = kE, kERO=kERO, kERC = kERC):
    jph = I*alpha*(1-beta*I)/(1+gamma*I)
    ksum = ko*O/kERO + kc*C/kERC
    fE = (1 + ((1/(kr*R))+1)*ksum + kr*R*(1+ksum)/(ko*O + kc*C))**(-1)
    jcA = kc*C*Et*(1+(kr*R/ksum))*fE
    joA = ko*O*Et*(1+(kr*R/ksum))*fE
    jpga = (2*jcA+1.5*joA)
    kEa = alphaa*kc
    kEb = alphab*kc
    jpc = (2/3)*jph
    jpr = (1/3)*jph
    jprod = (5/6)*((1/kEa)+(1/jpc)+(1/jpga)-(1/(jpga+jpc)))**(-1)
    jrubp = (1/kEb + 1/jpr + 1/jprod - 1/(jpr+jprod))**(-1)
    dRdt = -1*Et*fE*(kr*R+kc*C+ko*O) + jrubp
    return dRdt

def simforopt(alphaa, alphab, kr = kc, Ro = 100, tsteps = 10000, dt = .001, C = 230, O =210, kc = kc, ko = ko, I = I, Et = Et, alpha = alpha, beta = beta, gamma = gamma, kE = kE, kERO=kERO, kERC = kERC):
    R = np.zeros(tsteps)
    R[0] = (Ro)
    for t in range(tsteps-1):
        nex = R[t] + dRdtalt(R[t], alphaa, alphab)*dt
        R[t+1] = nex
        if R[t+1] > R[t]:
            R[tsteps-1] = -1
            print(t, " ", R[t+1])
            break
    
    maxi = R[tsteps-1]
    
    return maxi

pbounds = {'alphaa': (10, 20), 'alphab': (700, 800)}

optimizer = BayesianOptimization(
    f=simforopt,
    pbounds=pbounds,
    random_state=1,
)



optimizer.maximize(
    init_points=25,
    n_iter=50,
)
