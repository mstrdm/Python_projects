import numpy as np
import random

############################ STDP synapse #####################################
class STDP_synapse(object):
    def __init__(self, weight=0.5, dt=1e-3):
        self.dt = dt
        self.pre = 0.
        self.post = 0.
        self.tau_pre = 20e-3 #relaxation time
        self.tau_post = 80e-3
        self.s_pre = 0.
        self.s_post = 0.
        self.Ap = 2.5e-2 # Maximum increment during pairing (out of 1 total) 3e-2
        self.Ad = 0.8e-2 # Maximum decrement during pairing (out of 1) 0.8e-2
        self.weight = weight
        
    def change(self):
        self.weight += self.post*self.s_pre - self.pre*self.s_post 
        
        self.s_pre += self.pre*self.Ap - self.dt*self.s_pre/self.tau_pre
        self.s_post += self.post*self.Ad - self.dt*self.s_post/self.tau_post
        
        self.weight = max(self.weight,0)
        self.weight = min(self.weight,1)

############################ Dendritic synapse ################################
class dendr_synapse(object):
    def __init__(self, weight=0.5, dt=1e-3):
        self.dt = dt
        self.pre = 0.
        self.post = 0.
        
        # back-propagating spike parameters
        self.bth = 5#5e-3
        self.dumdb0 = 1#1e-3
        self.dumdb1 = 6#6e-3
        self.dumdb = self.dumdb0
        
        # dentritic potential parameters
        self.td = 20e-3 # dendritic potential relaxation time
        self.dumd = 100#20e-3 # originally 10e-3
        self.umdth = 10#10e-3
        self.umd = 0
        
        # synaptic weight parameters
        self.weight = weight
        self.wmax = 1 # originally 5
        self.wmin = 0.1 # originally 0.5
        self.dw = 0.01
        
    def change(self):
        self.dumdb = self.post*(self.dumdb0 + self.dumdb1/(1 + np.exp(-(self.umd-self.bth)/0.001))) # if postsynaptic neuron fired
        self.umd += self.weight*self.dumd*self.pre - self.dt*self.umd/self.td + self.dumdb        
        
        if self.post == 1:
            if self.umd >= self.umdth and self.weight < self.wmax:
                self.weight += (self.wmax-self.weight)*self.dw
                self.weight = min(self.weight, self.wmax)
            elif self.umd < self.umdth and self.weight > self.wmin:
                self.weight -= (self.weight-self.wmin)*self.dw
                self.weight = max(self.weight, self.wmin)
        
############################### LIF neuron ####################################
class LIF_neuron(object):
    def __init__(self, thr=10, tau=20e-3, dt=1e-3, refr=4e-3): #thr=4.5
        self.thr = thr
        self.dt = dt
        self.tau = tau
        self.psp = 0.
        self.input = 0.
        self.out = 0.
        self.refr = refr
        self.rest = 0
        
        self.t_mem = int(0.02/dt) # meaning: int(1second/dt)
        self.mem = np.zeros(self.t_mem)
        self.avg = 0.
            
    def fire(self):
#        self.psp = self.psp*np.exp(-1./self.tau) + (1.-self.out)*self.input
        if self.rest == 0:
            self.psp += self.input - self.dt*self.psp/self.tau
        else: self.rest -= 1
        self.out = 0. #Set Output back to zero if there was a spike previously
                       
        if self.psp > self.thr:
            self.out = 1.
            self.psp = 0.
            self.rest = int(self.refr/self.dt)
        if self.psp < 0:
            self.psp = 0
            
        # memory
        self.mem = np.append(self.mem, self.out)
        self.mem = self.mem[1:]
        self.avg = sum(self.mem)/(self.t_mem*self.dt)
        
########################## Periodic neuron ####################################
class periodic_neuron(object):
    def __init__(self, per, phase, dt): # phase (in seconds) is the delay until the first spike
        self.per = per
        self.phase = phase
        self.dt = dt # temporal resolution
        self.count = int(self.per/self.dt)
        self.out = 0
        self.time = 0
        
    def fire(self):
        if self.time > self.phase:
            if self.count >= int(self.per/self.dt):
                self.out = 1
                self.count = 1
            else:
                self.out = 0
                self.count += 1
        else:
            self.out = 0

        self.time += self.dt

###################### Poisson spike generator ################################
class poisson_neuron(object):
    def __init__(self, freq=50, dt=1e-3, refr=4e-3):
        self.dt = dt
        self.freq = freq
        self.refr = refr
        self.rest = 0
        self.out = 0.
    
    def fire(self):
        if self.rest == 0.:
            if self.freq*self.dt >= random.random():
                self.out = 1.
                self.rest = int(self.refr/self.dt)
            else: self.out = 0.
        else: 
            self.rest = max(0,self.rest-1)
            self.out = 0.
            
###############################################################################




































