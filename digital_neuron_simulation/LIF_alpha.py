## Script for simulating Spike Response Model LIF neuron behavior in FPGA.
## Several methods for implementing exponential function are coded here for comparison.
## Neuron object includes high precision (~analytical) solution for comparison.

import numpy as np
import matplotlib.pyplot as plt
import sys
import neural_classes

## Older class using higher precision for 0th bin and not using reciprocal values.
## This method requires digital divider circuits that take large area.
class exp_digi(object):
	def __init__(self):
		self.bins = 64									# number of bins for linearly fitting exp(-x)
		self.scale = self.bins/8						# amount of horizontal stretching for the exp(-x) function
		self.round_k_LP = 4								# upscaling factor for better rounding results (low precision - exponent tail)
		self.round_k_HP = 4								# ... (high precision - exponent head)
		self.max = 4096									# maximum value for resulting exp(-x) function
		self.x_tot_LP = 2048							# amount of argument values in valid exp(-x) range (over low precision range)
		self.x_tot_HP = 512								# ... (over high precision range)
		self.dx = int(self.x_tot_LP/self.bins)			# amount of low precision argument values per bin
		self.fit_params = np.zeros((self.bins, 2))		# stores a and b parameters for fitting exp in each bin (exp = a*x + b)

		## Constructing exponential function fitting parameter table
		for i in range(self.bins):
			if i == 0:
				self.fit_params[i,0] = self.max*( 1 - np.exp(-1/self.scale) )/self.x_tot_HP
				self.round_k = self.round_k_HP
			else:
				self.fit_params[i,0] = self.max*( np.exp(-i/self.scale) - np.exp(-(i+1)/self.scale) )/self.dx
				self.round_k = self.round_k_LP

			self.fit_params[i,0] = round(self.round_k*self.fit_params[i,0])
			self.fit_params[i,1] = self.round_k*self.max*np.exp(-i/self.scale) + i*self.fit_params[i,0]*self.dx
			self.fit_params[i,1] = round(self.fit_params[i,1]/self.round_k)			

	def exp_(self, t):
		bin_n = min(int(-t*self.scale), self.bins-1)	# bin number

		if bin_n == 0:
														# "stretched" exponent argument:
			t0 = min(int(-t*self.scale*self.x_tot_HP), self.x_tot_HP-1)				
			self.round_k = self.round_k_HP
		else:
			t0 = min(int(-t*self.scale*self.x_tot_LP/self.bins), self.x_tot_LP-1)
			self.round_k = self.round_k_LP

		### EXP calculation using universal LUT:
		# exp_val = int((-self.fit_params[bin_n,0]*t0 + self.round_k*self.fit_params[bin_n,1])/self.round_k)

		### EXP calculation usign single LUT adapted for particular tau:
		exp_val = int(self.max*np.exp(t))

		return exp_val

# Newer class that doesn't require binary division, uses reciprocal LUT for evaluating 1/tau.
# This method uses faster and smaller digital circuits than the older one.
class exp_digi_rec(object):
	def __init__(self):
		self.bins = 64									# number of bins for linearly fitting exp(-x)
		self.scale = 2**8								# targeting exp(-2^18/scale) = exp(-8)
		self.round_k = 16								# upscaling factor for better rounding results
		self.max = 2**8-1								# exponent value at exp(0)
		self.x_tot = 2**11								# total number of points from exp(0) to exp(-8)
		self.dx = int(self.x_tot/self.bins)				# number of points per bin
		self.fit_params = np.zeros((self.bins, 2))		# stores a and b parameters for fitting exp in each bin (exp = a*x + b)
		self.f_upsc = 2**12-1							# upscaling facrot used for reciprocal LUT
		self.denum_max = 64								# maximum denumenator value in exp(-num/denum)
		self.recip_lut = np.zeros(self.denum_max)		# reciprocal LUT

		## Constructing exponential function fitting parameter table
		for i in range(self.bins):
			self.fit_params[i,0] = self.max*( np.exp(-i*self.dx/self.scale) - np.exp(-(i+1)*self.dx/self.scale) )/self.dx
			self.fit_params[i,0] = round(self.round_k*self.fit_params[i,0])

			self.fit_params[i,1] = self.round_k*self.max*np.exp(-i*self.dx/self.scale) + i*self.fit_params[i,0]*self.dx
			self.fit_params[i,1] = round(self.fit_params[i,1]/self.round_k)		

		## Constructing table for reciprocal values
		for i in range(1, self.denum_max):
			self.recip_lut[i] = round(self.f_upsc/i)

	def exp_(self, tn, td):
		td = int(td)
		t = -tn*self.recip_lut[td]						# calculating exp argument (exp(-t))
		t = min(int(t*self.scale/self.f_upsc), self.x_tot-1)				# scaling t to match the scale used in LUT

		bin_n = int(t/self.dx)
		exp_val = int((-self.fit_params[bin_n,0]*t + self.round_k*self.fit_params[bin_n,1])/self.round_k)

		return exp_val

# Class for implementing exp by using two LUTs, one storing max values for each bin and the other - exp template withing the bin.
# Requires slightly more FPGA resources, but works slightly faster than the linear fit implementation.
class exp_digi_luts(object):
	def __init__(self):
		self.bins = 2**6								# number of bins for dividing exponent range
		self.bin_size = 2**6							# number of points per bin
		self.x_max = 8									# range of exp(-x) function (maximum x value)
		self.bin_lut = np.zeros(self.bins)				# lut for max bin values
		self.exp_lut = np.zeros(self.bin_size)			# lut for exp template
														# argument scaling factor (LUT for exp(-x/scale) will be stored):
		self.scale = (self.bins*self.bin_size/self.x_max)
		self.max = 255									# maximum exp value in LUTs
		self.f_upsc = 2**12								# upscaling facrot used for reciprocal LUT
		self.denum_max = 64								# maximum denumenator value in exp(-num/denum)
		self.recip_lut = np.zeros(self.denum_max)		# reciprocal LUT

		## Constructing exponent and bin LUTs
		for i in range(self.bins):
			self.bin_lut[i] = round(self.max*np.exp(-i*self.bin_size/self.scale))
		for i in range(self.bin_size):
			self.exp_lut[i] = round(self.max*np.exp(-i/self.scale))

		## Constructing table for reciprocal values
		for i in range(1, self.denum_max):
			self.recip_lut[i] = round(self.f_upsc/i)

	def exp_(self, tn, td):
		td = int(td)
		t = -tn*self.recip_lut[td]						# calculating exp argument (exp(-t))
														# scaling t to match the scale used in LUTs:
		t = min(int(t*self.scale/self.f_upsc), self.bins*self.bin_size-1)	
		bin_n = int(t/self.bins)#min(self.bins-1, int(i/bins))
		exp_n = t%self.bins
		exp_val = int(self.bin_lut[bin_n]*self.exp_lut[exp_n]/self.max)

		return exp_val

class LIF_alpha(object):
	def __init__(self, dt=1e-3, tau_psc=40e-3, tau_psp=10e-3, refr=4e-3, thr=1, kw_n=1, kw_d=1):
		# self.exp_digi = exp_digi()						# digital exponential function simulator
		# self.exp_digi = exp_digi_rec()
		self.exp_digi = exp_digi_luts()

		### LIF parameters ###
		self.dt = dt
		self.tau_psc = int(tau_psc/self.dt)
		self.tau_psp = int(tau_psp/self.dt)
		self.refr = int(refr/self.dt)

		### Scaling factors and derivative parameters ###
		self.kw_n = kw_n								# weight scaling numenator
		self.kw_d = kw_d								# weight scaling denumenator
		self.exp_max = self.exp_digi.max 				# exponential function peak value
														# threshold value stored in RAM:
		self.thr_stored = round(thr*self.kw_n/self.kw_d)			
														# threshold upscaling factor:
		self.f = (self.tau_psc-self.tau_psp)*self.exp_max*self.kw_n/(self.kw_d*self.tau_psc)
		self.thr = round(self.thr_stored*self.f)		# threshold value upscaled to match PSP values
		self.t_max = 63									# maximum time value

		### Variables stored in RAM ###
		self.exp1_amp = 0
		self.exp2_amp = 0
		self.t = 0

		### Derivative variable values ###
		self.exp1 = 0
		self.exp2 = 0
		self.psp = 0
		self.out = 0

		### Input variables ###
		self.input = 0

		### Auxiliary variables ###
		self.refr_count = 0

		### Real (not rounded) values for comparison
		self.exp1_real = 0
		self.exp2_real = 0
		self.psp_real = 0
		self.exp1_amp_real = 0
		self.exp2_amp_real = 0

	def step(self):
		## Deasserting output after a spike:
		if self.out == 1:
			self.out = 0

		## Ignoring neuronal dynamics until refractory period ends:
		if self.refr_count > 0:
			self.refr_count -= 1
		## Neuron dynamics
		else:
			## Looping the time upon it reaching t_max:
			if self.t == self.t_max:
				self.exp1_amp = round(self.exp1_amp*self.exp_digi.exp_(-self.t,self.tau_psc)/self.exp_max)
				self.exp2_amp = round(self.exp2_amp*self.exp_digi.exp_(-self.t,self.tau_psp)/self.exp_max)
				self.exp1_amp_real = self.exp1_amp_real*np.exp(-self.t/self.tau_psc)
				self.exp2_amp_real = self.exp2_amp_real*np.exp(-self.t/self.tau_psp)
				self.t = 0
				# self.exp1 = 0
				# self.exp2 = 0
			else:
				self.exp1 = self.exp1_amp*self.exp_digi.exp_(-self.t,self.tau_psc) # use (-self.t/self.tau_psc) for older exp class
				self.exp2 = self.exp2_amp*self.exp_digi.exp_(-self.t,self.tau_psp)
			self.psp = self.exp1 - self.exp2

			## Real value calculation:
			self.exp1_real = self.exp1_amp_real*self.exp_max*np.exp(-self.t/self.tau_psc)
			self.exp2_real = self.exp2_amp_real*self.exp_max*np.exp(-self.t/self.tau_psp)
			self.psp_real = self.exp1_real - self.exp2_real

			## Neuron firing:
			if self.psp >= self.thr:
				self.out = 1
				self.refr_count = self.refr
				## Resets
				self.t = 0
				self.exp1_amp = 0
				self.exp2_amp = 0
				self.psp = 0
				self.exp1_amp_real = 0
				self.exp2_amp_real = 0
				self.psp_real = 0

			## Reading neuron input (only if neuron didn't fire):
			else:
				if self.input != 0:
					self.t = 0

					self.exp1_amp = self.exp1/self.exp_max + round(self.input*self.kw_n/self.kw_d)
					self.exp2_amp = self.exp2/self.exp_max + round(self.input*self.kw_n/self.kw_d)

					## Rounding amplitudes to be stored in RAM:
					self.exp1_amp = round(self.exp1_amp)
					self.exp2_amp = round(self.exp2_amp)

					print(self.exp1_amp, self.exp2_amp)

					## Real value calculation:
					self.exp1_amp_real = self.exp1_real/self.exp_max + round(self.input*self.kw_n/self.kw_d)
					self.exp2_amp_real = self.exp2_real/self.exp_max + round(self.input*self.kw_n/self.kw_d)

				# self.t = min(self.t+1, 255)
				self.t += 1

## Neuron simulation:
tmax = 1
dt = 1e-3
T = int(tmax/dt)

refr = 4e-3
tau_psc = 40e-3
tau_psp = 10e-3
w_in = 2
thr = 100000*w_in*2			# <---- CHANGE TO MAKE NEURON FIRE
kw_n = 1
kw_d = 1

lif = LIF_alpha(dt=1e-3, tau_psc=tau_psc, tau_psp=tau_psp, refr=refr, thr=thr, kw_n=kw_n, kw_d=kw_d)
poisson = neural_classes.poisson_neuron(freq=50, dt=dt, refr=0e-3)

output = np.zeros(T)
psp = np.zeros((T,3))
psp_real = np.zeros((T,3))

for t in range(T):
	lif.step()
	output[t] = lif.out
	psp[t,0] = lif.exp1*kw_d/kw_n/lif.f/w_in
	psp[t,1] = lif.exp2*kw_d/kw_n/lif.f/w_in
	psp[t,2] = lif.psp*kw_d/kw_n/lif.f/w_in
	psp_real[t,0] = lif.exp1_real*kw_d/kw_n/lif.f/w_in
	psp_real[t,1] = lif.exp2_real*kw_d/kw_n/lif.f/w_in
	psp_real[t,2] = lif.psp_real*kw_d/kw_n/lif.f/w_in

	poisson.fire()
	lif.input = round(w_in*kw_n/kw_d)*poisson.out

plt.figure(figsize=(6,6))
plt.subplot(311)
plt.plot(psp, ls='steps')
plt.subplot(312)
plt.plot(output, ls='steps')
plt.subplot(313)
plt.plot(psp[:,2], ls='steps')
plt.plot(psp_real[:,2])
plt.show()
