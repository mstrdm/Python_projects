## Controller for programming, stimulating and reading out neuromrophic system through UART.
## Stimulation uses temporal pattern stored in the seq.txt file, which is assumed to be melody. The output of the system is converted to audio.
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.multiprocess as mp
import serial
import threading
from collections import OrderedDict
import time
import pygame.midi

# ----------------------------------------------------------------------------------------------------------------------------
class EventPlotter(object):
	def __init__(self, ser, player):
		self.app = pg.mkQApp()
		self.proc = mp.QtProcess()
		self.rpg = self.proc._import('pyqtgraph')
				
		self.plotwin = self.rpg.GraphicsWindow(title="Monitor")
		# self.plotwin = pg.GraphicsWindow(title="Monitor")
		self.plotwin.resize(1000,600)
		self.plotwin.setWindowTitle('Activity Monitor')
		self.p1 = self.plotwin.addPlot(title="Neuron spikes vs. time")
		self.p1.setLabel('left', 'Neuron Id')
		self.p1.setLabel('bottom', 'Time [s]')
		self.p1.showGrid(x=True, y=True, alpha=0.5)
		self.spikes_curve = self.p1.plot(pen=None, symbol="o", symbolPen=None, symbolBrush='w', symbolSize=3)   

		# self.app.exit(self.app.exec_()) # not sure if this is necessary
		
		self.on_screen = 4000 # Number of events on the screen
		self.all_time_stamps = np.zeros(self.on_screen)
		self.all_addresses = np.zeros(self.on_screen, dtype=int)
		
		self.ser = ser
		self.player = player

		self.old_stamp = 0

		# For evaluating accuracy
		self.acc_eval = False
		self.out_stamp = 0
		self.last_in = 0
		self.in_rec = 0
		self.last_out = 0
		self.acc_total = -1
		self.acc_corr = 0
		
	def decode_events(self, byte_data):
		time_stamps = []
		addresses = []
		event_nr = int(len(byte_data)/3)
		
		if event_nr > 0:
			for e in range(event_nr):
				event = byte_data[e*3:e*3+3]
				addresses.append(event[2])
				new_stamp = int.from_bytes(event[0:2], byteorder='big')

				# Accuracy evaluation:
				if self.acc_eval == True:
					if event[2] >= 20 and event[2] < 40:
						if new_stamp - self.out_stamp > 190 or new_stamp < self.out_stamp:
							self.acc_total += 1
							if self.acc_total > 0:
								self.acc_corr += self.last_out == self.last_in
							self.p1.setTitle(title='Accuracy = ' + str(round(100*self.acc_corr/self.acc_total, 2)) + '%')
							self.last_out = event[2] - 20
							self.last_in = self.in_rec
							self.out_stamp = new_stamp

					elif event[2] < 20:
						self.last_in = event[2]

				# Playing notes:
				if event[2] > 20 and event[2] < 40:								# keep silent on 0th output neuron firing (20th neuron in the graph)
					if new_stamp - self.old_stamp > 190 or new_stamp < self.old_stamp:
						self.player.note_on(48+event[2], 100)
						self.old_stamp = new_stamp
					
				time_stamps.append(new_stamp)
				
		return time_stamps, addresses

	def ReadEvents(self):
		try:
			event_data = self.ser.read(30)
			time_stamps, addresses = self.decode_events(event_data)
			dn = len(time_stamps)
			if dn > 0:
				self.all_time_stamps = np.roll(self.all_time_stamps, -dn)
				self.all_addresses = np.roll(self.all_addresses, -dn)
				self.all_time_stamps[-dn:] = np.array(time_stamps)
				self.all_addresses[-dn:] = np.array(addresses)
				
				self.spikes_curve.setData(x=self.all_time_stamps, y=self.all_addresses, _callSync='off')
				
		except:
			None

# ----------------------------------------------------------------------------------------------------------------------------
class parameters(object):
	def __init__(self, ser):
		self.n_param = 25		# total number of parameters (1 - firing rate, 12 - hidden layer, 12 - output layer)
		self.ser = ser

		self.curval_dic = OrderedDict()
		self.curval_dic['IN_PER'] = 0

		self.curval_dic['HL_TAU_DP'] = 0
		self.curval_dic['HL_TAU_PSC'] = 0
		self.curval_dic['HL_TAU_PSP'] = 0
		self.curval_dic['HL_DW_POS'] = 0
		self.curval_dic['HL_DW_NEG'] = 0
		self.curval_dic['HL_DP_THR'] = 0
		self.curval_dic['HL_THR'] = 0
		self.curval_dic['HL_DP_THR_MIN'] = 0
		self.curval_dic['HL_W_MAX'] = 0
		self.curval_dic['HL_W_INH'] = 0
		self.curval_dic['HL_W_SUP'] = 0
		self.curval_dic['HL_W_CONST'] = 0

		self.curval_dic['OL_TAU_DP'] = 0
		self.curval_dic['OL_TAU_PSC'] = 0
		self.curval_dic['OL_TAU_PSP'] = 0
		self.curval_dic['OL_DW_POS'] = 0
		self.curval_dic['OL_DW_NEG'] = 0
		self.curval_dic['OL_DP_THR'] = 0
		self.curval_dic['OL_THR'] = 0
		self.curval_dic['OL_DP_THR_MIN'] = 0
		self.curval_dic['OL_W_MAX'] = 0
		self.curval_dic['OL_W_INH'] = 0
		self.curval_dic['OL_W_SUP'] = 0
		self.curval_dic['OL_W_CONST'] = 0

	def load(self, fname='param.ini'):
		error_count = 0
		msg_on = True
		with open(fname, 'r') as param_file:
			for line in param_file:
				if line[0] != '#':
					curline = line.split()

					if len(curline) != 0:
						try:
							self.curval_dic[curline[0]] = int(curline[2])
						except ValueError:
							if msg_on:
								error_count += 1
								print('Unknown parameter found in ' + fname + '. Ignoring...')
						except IndexError:
							if msg_on:
								error_count += 1
								print('Invalid line is present in ' + fname + ' or wrong syntax is used for parameter definition. Ignoring...')
						if error_count > 9 and msg_on:
							msg_on = False
							print("Too many errors. Is that the correct file? Will ignore all further errors without notice.")		

	def write(self, fname='param.ini'):
		t0 = time.time()
		self.ser.timeout = 0.1
		self.load(fname=fname)
		self.ser.write(bytes.fromhex('02'))
		for item in self.curval_dic.items():
			byte_data = int(item[1]).to_bytes(2, byteorder='big')
			self.ser.write(byte_data)
		# self.ser.timeout = 0
		print("Writing done in", time.time()-t0, "seconds.\n")

	def read(self):
		t0 = time.time()
		self.ser.timeout = 0.1
		self.ser.write(bytes.fromhex('01'))
		parameters = []
		while True:
			byte_data = ser.read(2)
			if len(byte_data) == 0:
				break
			else:
				parameters.append(int.from_bytes(byte_data, byteorder='big'))

		if len(parameters) == self.n_param:
			k = 0
			print("---- SYSTEM PARAMETER VALUES ----\n")
			for item in self.curval_dic.items():
				print(item[0], '=', parameters[k])
				k += 1
			print("\nReading done in", time.time()-t0, "seconds.\n")
		else:
			print('ERROR: Wrong number of parameters.\n')
		# self.ser.timeout = 0

# ----------------------------------------------------------------------------------------------------------------------------
class weights(object):
	def __init__(self, ser):
		self.n_in = 20					# number of inputs
		self.n_del = 8					# number of delay nodes in input group
		self.n_hl = 200					# number of hidden layer neurons
		self.plot_weights = True
		self.ser = ser

	def write(self, fname='syn_ini.txt'):
		t0 = time.time()
		self.ser.timeout = 0.1
		self.ser.write(bytes.fromhex('06'))
		f = np.flip(np.loadtxt(fname, dtype=int))

		for i in range(len(f)):
			byte_data = int(f[i]).to_bytes(2, byteorder='big')
			self.ser.write(byte_data)
		# self.ser.timeout = 0
		print("Writing done in", time.time()-t0, "seconds.\n")

	def plot(self, weights):
		plt.figure(figsize=(6,6))
		plt.subplot(211)
		plt.plot(weights[0:self.n_hl*self.n_in], ls='steps')
		plt.subplot(212)
		plt.plot(weights[self.n_hl*self.n_in:], ls='steps')
		plt.show()

	def read(self, fname='read_weights.txt'):
		t0 = time.time()
		self.ser.timeout = 0.1
		self.ser.write(bytes.fromhex('05'))
		weights = []
		while True:
			byte_data = ser.read(2)
			if len(byte_data) == 0:
				break
			else:
				weights.append(int.from_bytes(byte_data, byteorder='little'))

		if len(weights) == self.n_in*self.n_del*self.n_hl + self.n_in*self.n_hl:
			np.savetxt(fname, np.flip(np.array(weights)), fmt='%i')
			print("\nReading done in", time.time()-t0, "seconds.\nPLEASE CLOSE THE PLOT WINDOW TO CONTINUE.\n")
		else:
			print('ERROR: Wrong number of weights.\n')
		# self.ser.timeout = 0

		if self.plot_weights:
			self.plot(np.flip(np.array(weights)))			

# ----------------------------------------------------------------------------------------------------------------------------

## ---- CODE ----

script_on = True
spikes_on = False
in_driver_en = False

def send_cmd():
	global spikes_on, script_on, in_driver_en, ser, player, EventPlotter
	while True:
		cmd = input('cmd?: ')
		if cmd == 'go':
			ser.write(bytes.fromhex('07'))		# go to spikes state
			ser.read(1000)
			ser.timeout = 0.5
			spikes_on = True
			time.sleep(100e-3)

		if cmd == 'learn':
			if spikes_on == True:
				EventPlotter.acc_eval = False
				in_driver_en = False
				ser.write(bytes.fromhex('FF'))
				ser.write(bytes.fromhex('0' + hex(int('011', 2))[2]))
				in_driver_en = True
			else:
				print('Current state is idle. Enter "go" to go to spikes_on state.')

		if cmd == 'recall':
			if spikes_on == True:
				EventPlotter.acc_eval = False
				in_driver_en = False
				ser.write(bytes.fromhex('FF'))
				ser.write(bytes.fromhex('0' + hex(int('110', 2))[2]))
			else:
				print('Current state is idle. Enter "go" to go to spikes_on state.')

		if cmd == 'follow':
			if spikes_on == True:
				EventPlotter.acc_eval = True
				EventPlotter.acc_total = -1
				EventPlotter.acc_corr = 0
				in_driver_en = False
				ser.write(bytes.fromhex('FF'))
				ser.write(bytes.fromhex('0' + hex(int('010', 2))[2]))
				in_driver_en = True
			else:
				print('Current state is idle. Enter "go" to go to spikes_on state.')				

		if cmd == 'pause':
			if spikes_on == True:
				EventPlotter.acc_eval = False
				in_driver_en = False
				ser.write(bytes.fromhex('FF'))
				ser.write(bytes.fromhex('0' + hex(int('000', 2))[2]))
			else:
				print('Current state is idle. Enter "go" to go to spikes_on state.')

		if cmd == 'stop':
			EventPlotter.acc_eval = False
			ser.timeout = 0.1
			spikes_on = False
			in_driver_en = False
			time.sleep(100e-3)
			ser.write(bytes.fromhex('FF'))		# interrupt
			ser.write(bytes.fromhex('FF'))		# go to idle state
			ser.read(1000)

		if cmd == 'rw':
			if spikes_on == False:
				WeightReadWrite.read('syn_read.txt')
			else:
				print('Command unavailable in spikes_on state. Enter "stop" to go to idle state.')

		if cmd == 'ww':
			if spikes_on == False:
				WeightReadWrite.write('syn_ini.txt')
			else:
				print('Command unavailable in spikes_on state. Enter "stop" to go to idle state.')

		if cmd == 'rp':
			if spikes_on == False:
				ParamReadWrite.read()
			else:
				print('Command unavailable in spikes_on state. Enter "stop" to go to idle state.')

		if cmd == 'wp':
			if spikes_on == False:
				ParamReadWrite.write('param.ini')
			else:
				print('Command unavailable in spikes_on state. Enter "stop" to go to idle state.')

		if cmd == 'clear':
			EventPlotter.all_time_stamps = np.zeros(EventPlotter.on_screen)
			EventPlotter.all_addresses = np.zeros(EventPlotter.on_screen, dtype = int)

		if cmd == 'quit':
			spikes_on = False
			script_on = False
			time.sleep(110e-3)
			ser.close()
			del player
			pygame.midi.quit()
			break

def run_input(fname='seq.txt'):
	global spikes_on, ser, script_on, in_driver_en, EventPlotter

	while script_on == True:
		time.sleep(100e-3)
		if spikes_on == True:
			seq = np.loadtxt(fname)
			ser.write(int(seq[0]).to_bytes(1, byteorder='big'))
			# k_max = len(seq) - 1
			while in_driver_en == False:
				time.sleep(100e-3)
				if spikes_on == False:
					break

			k = 0
			while spikes_on == True:
				# change input delay here:
				time.sleep(200e-3)
				if in_driver_en == True:
					k += 1
					if k == len(seq):
						k = 0
					ser.write(int(seq[k]).to_bytes(1, byteorder='big'))

def run_plot():
	global spikes_on, script_on, EventPlotter
	while script_on == True:
		time.sleep(100e-3)
		if spikes_on == True:
			EventPlotter.ReadEvents()
	EventPlotter.proc.close()

# Serial Port Initialization
ser = serial.Serial()
ser.baudrate = 115200#460800
ser.port = 'COM5'
ser.timeout = 0.1
ser.open()

# MIDI Port Initialization
pygame.midi.init()
player = pygame.midi.Output(0)
player.set_instrument(0)

player.note_on(50, 100)

# Read/Write Object Initialization
EventPlotter = EventPlotter(ser=ser, player=player)
ParamReadWrite = parameters(ser=ser)
WeightReadWrite = weights(ser=ser)

# Creating separate threads for simultaneously running input and plotting
thread_input = threading.Thread(target=run_input)
thread_input.daemon = False
thread_input.start()

# thread_player = threading.Thread(target=play_note)
# thread_player.daemon = False
# thread_player.start()

thread_plot = threading.Thread(target=run_plot)
thread_plot.daemon = False
thread_plot.start()

send_cmd()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	