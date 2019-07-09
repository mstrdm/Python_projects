## Standalone script for generating spikes from different images (several classes available) and applying them to neuromorphic chip through USB.
## Feed-forward topology is assumed; synaptic weights to each output neuron are periodically read and displayed.
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.multiprocess as mp
import usb1
import time
import binascii
import bitstring
import sys
sys.path.append('lib')
import rw_lib
import random
import threading
from PIL import Image
import struct

#############################################################################################################

class ImageSpikeGenerator(object):
	def __init__(self, max_activity=50, hop=25):
		## Auxiliary and neuron parameters
		self.max_activity = max_activity # maximum activity of the neurons. Used for normalizing compressed image
		self.hop = hop # normally 200
		self.dt = 1e-3
		self.refr_per = 4e-3
		
		# self.img_width = 20
		self.img_size = (20,20)
		self.handle = handle

		self.count = self.hop
		self.rates = np.zeros(self.img_size) + 1e-6
		self.refr = np.zeros(self.img_size[0]*self.img_size[1])

		## Parameters for draw_object
		self.x, self.y, self.dx, self.dy = 0, 0, 0, 0
		self.move = 0

		## Plot initialization
		self.pos = np.array([0.0, 0.5, 1.0])
		self.color = np.array([[29,82,135,255], [53,193,133,255], [208,246,106,255]])
		self.color_map = pg.ColorMap(self.pos, self.color, mode=None)

		self.win = pg.image(np.zeros((self.img_size)) + 1e-6)
		self.win.view.setAspectLocked(False)
		self.win.setColorMap(self.color_map)
		self.win.resize(170,170)
		self.win.ui.histogram.hide()
		self.win.ui.roiBtn.hide()
		self.win.ui.menuBtn.hide()

	def load_DVS(self):
		import socket
		host, port ='127.0.0.1', 7777
		self.buf_size = 63000
		self.DVS_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.DVS_socket.bind((host, port))

		self.xmask = 0x00FE ## 0111_1111_0000_0000
		self.xshift = 1
		self.ymask = 0x7F00 ## 0000_0000_1111_1110
		self.yshift = 8
		self.pmask = 0x1 ## 0000_0000_0000_0001
		self.pshift = 0

		self.DVS_thr = 300

		self.DVS_filters = np.zeros(self.img_size)

	def draw_DVS(self): ## this function is stand-alone and does not require calling NextFrame(). Pass its output directly to matrix_converter
		spike_matrix = np.zeros(self.img_size, dtype=int)
		dvs_data = self.DVS_socket.recv(self.buf_size)

		count = 4
		while count < len(dvs_data):
			addr = struct.unpack('>I', dvs_data[count:count+4])[0]
			count += 8

			x_addr = (addr & self.xmask) >> self.xshift
			y_addr = (addr & self.ymask) >> self.yshift

			neur_x = int(x_addr*self.img_size[1]/130)
			neur_y = int(y_addr*self.img_size[0]/130)

			self.DVS_filters[neur_y, neur_x] += 1
			if self.DVS_filters[neur_y, neur_x] >= self.DVS_thr:
				self.DVS_filters[neur_y, neur_x] = 0
				spike_matrix[neur_y, neur_x] = 1

		self.win.setImage(np.transpose(spike_matrix)+1e-6, autoRange=True)
		pg.QtGui.QApplication.processEvents()
		return spike_matrix

	def load_wav(self, path='', t_disp=100e-3): # initialization function for draw_wave function
		from scipy.io import wavfile
		fs, self.wav_data = wavfile.read(path)
		self.samp_disp = int(fs*t_disp)										# samples to display
		self.samp_pix = int(self.samp_disp/self.img_size[1])				# samples per image pixel
		self.samp_avg = int(np.ceil(0.5*self.samp_disp/self.img_size[0]))	# FFT samples to average per image pixel
		self.wav_data = self.wav_data[:,0]
		self.wav_ptr = 0
		self.fft_frame = np.zeros(self.img_size)

	def draw_wave(self, FFT=True):
		wav_ptr_nxt = self.wav_ptr + self.samp_disp
		if wav_ptr_nxt > len(self.wav_data):
			self.wav_ptr = 0
			wav_ptr_nxt = self.samp_disp
		frame_data = self.wav_data[self.wav_ptr:wav_ptr_nxt]
		self.wav_ptr = wav_ptr_nxt

		if FFT == False:
			frame_data_avg = np.zeros(self.img_size[1])
			for i in range(self.img_size[1]):
				frame_data_avg[i] = np.sum(frame_data[i*self.samp_pix:(i+1)*self.samp_pix])/self.samp_pix
				# frame_data_avg[i] = int(np.sum(frame_data[i*self.samp_pix:(i+1)*self.samp_pix])/self.samp_pix)

			amp = np.amax([np.amax(frame_data_avg), np.abs(np.amin(frame_data_avg))])
			scale = (int(self.img_size[0]/2)-1)/amp
			frame_data_avg = frame_data_avg*scale + int(self.img_size[0]/2)

			image = np.zeros(self.img_size)
			for i in range(self.img_size[1]):
				image[int(frame_data_avg[i]), i] = 1

		elif FFT == True:
			frame_fft = np.abs(np.fft.rfft(frame_data))
			frame_fft.resize(self.img_size[0]*self.samp_avg)

			frame_fft_avg = np.zeros(self.img_size[0])
			for ii in range(self.img_size[0]):
				frame_fft_avg[ii] = np.mean(frame_fft[ii*self.samp_avg:(ii+1)*self.samp_avg])

			self.fft_frame = np.roll(self.fft_frame, -1)
			self.fft_frame[:, self.img_size[1]-1] = frame_fft_avg

			## Normalization (comment this line if Log is used below)
			image = self.fft_frame/np.max(self.fft_frame)

			# ## Log
			# image = np.log(self.fft_frame + 1e-6)
			# image += np.min(image)
			# image = image/np.max(image)

		return self.max_activity*image

	def draw_img(self, path=''):
		img = Image.open(path).convert('L') # loading image and converting to greyscale
		img = img.resize(self.img_size, Image.ANTIALIAS) # downscaling
		img = img.transpose(Image.TRANSPOSE)
		image = np.array(img) # converting to numpy array
		image = self.max_activity*(1 - abs(image/np.max(image))) # normalization
		return image

	def draw_bar(self, alpha=45, width=100, thick=15, noise=[0,0], compress=True): # compress downsizes and normalizes the image
		crop_per = (int(width/self.img_size[0]), int(width/self.img_size[1]))
		alpha = np.deg2rad(alpha) # bar angle is from 0 to 179
		center_y = int(width/2 - 1 + noise[0]*width*random.random()) # coordinate of the center
		center_x = int(width/2 - 1 + noise[1]*width*random.random())        
		hor_thick = min(thick/np.sin(alpha), (thick+1)*width) # horizontal bar thickness under given angle
		delta_c = min(1/np.tan(alpha), width) # center displacement per vertical pixel
		image = np.zeros((width, width), dtype=int)

		for row in range(width):
			vert_disp = row - center_y # vertical displacement from the center
			loc_center = center_x + delta_c*vert_disp # local bar center at this row
			start_x = int(max(loc_center - int(hor_thick/2), 0))
			end_x = int(min(loc_center + int(hor_thick/2), width))
			
			if end_x >= 0:
				image[row][start_x:end_x] = 1
				
		for x in range(width): # Circular mask
			for y in range(width):
				if np.sqrt((x-center_x)**2+(y-center_y)**2) > width/2.:
					image[y,x] = 0
					
		if compress == True: # Compression 
			compr_image = np.zeros(self.img_size) # compressed image
			for x in range(self.img_size[1]):
				for y in range(self.img_size[0]):
					compr_image[y,x] = np.sum(image[y*crop_per[0]:(y+1)*crop_per[0], x*crop_per[1]:(x+1)*crop_per[1]])
			return self.max_activity*compr_image/np.amax(compr_image)
		else:
			return image
	
	def draw_particle(self, dx0=3, rad=2, move_int=50):
		image = np.zeros(self.img_size)

		if self.move == 0:
			self.move = move_int
			alpha = np.deg2rad(90*random.random())
			self.dy = random.choice([-1,1])*int(dx0*np.sin(alpha))
			self.dx = random.choice([-1,1])*int(dx0*np.cos(alpha))
		else:
			self.move -= 1              

		self.x += self.dx
		self.y += self.dy
		x_draw, y_draw = 0, 0
		for x in range(self.x-rad, self.x+rad+1):
			for y in range(self.y-rad, self.y+rad+1):
				if np.sqrt((x-self.x)**2+(y-self.y)**2) <= rad:
					if x >= self.img_size[1]:
					   x_draw = x - self.img_size[1]
					elif x < 0:
					   x_draw = x + self.img_size[1]
					else: x_draw = x

					if y >= self.img_size[0]:
					   y_draw = y - self.img_size[0]
					elif y < 0:
					   y_draw = y + self.img_size[0]
					else: y_draw = y
					image[y_draw, x_draw] = 1

		if self.x >= self.img_size[1]:
			self.x -= self.img_size[1]
		if self.x < 0:
			self.x += self.img_size[1]
		if self.y >= self.img_size[0]:
			self.y -= self.img_size[0]  
		if self.y < 0:
			self.y += self.img_size[0]   

		return self.max_activity*image

	def NextFrame(self, func='draw_bar', img_path=''):
		spike_matrix = np.zeros((self.img_size[0]*self.img_size[1], 2), dtype=int)

		if self.count == self.hop:
		#time.sleep(10e-3)
			if func == 'draw_bar':
				self.rates = self.draw_bar(alpha=179*random.random(), compress=True)
			elif func == 'draw_particle':
				self.rates = self.draw_particle()
			elif func == 'draw_img':
				self.rates = self.draw_img(path=img_path)
			elif func == 'draw_wave':
				self.rates = self.draw_wave(FFT=False)

			self.win.setImage(np.transpose(self.rates), autoRange=True)
			pg.QtGui.QApplication.processEvents()

			self.rates = self.rates.flatten()

			self.count = 0
		else: self.count += 1
	
		for i in range(self.img_size[0]*self.img_size[1]):

			if self.rates[i]*self.dt >= random.random() and self.refr[i] == 0:
				spike_matrix[i,0] = 1
				self.refr[i] = self.refr_per
			else:
				self.refr[i] = max(self.refr[i]-self.dt, 0)

		return spike_matrix

#############################################################################################################

class usbEventSender(object):
	def __init__(self, handle):
		self.handle = handle

	def inflate_str(self, data_str=''):
		new_str = ''
		for i in range(len(data_str)):
			new_str += data_str[i] + ' '
		return new_str

	def matrix_converter(self, spike_matrix):
		bin_str = ''
		for i in range(spike_matrix.shape[0]):
			for ii in range(spike_matrix.shape[1]):
				if spike_matrix[i,ii] == 1:
					adr = spike_matrix.shape[0]*ii+i
					adr_bin = ('{0:0' + str(32) + 'b}').format(int(adr+1)) # Here I add 1 to the address to remove the '0' address
					bin_str += adr_bin

		bin_data = self.inflate_str(bin_str)
		byte_data = bytearray(np.packbits(np.fromstring(bin_data, sep=' ', dtype=int)))
		return byte_data

	def SendEvents(self, spike_matrix=np.zeros((10,2))):
		if sum(sum(spike_matrix)) > 0:
			byte_data = self.matrix_converter(spike_matrix)
			self.handle.bulkWrite(endpoint=0x01, data=byte_data, timeout=1)

#############################################################################################################

class EventPlotter(object):
	def __init__(self, handle):
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

		self.last_event = 0
		
		self.on_screen = 10000 # Number of events on the screen
		self.all_time_stamps = np.zeros(self.on_screen)
		self.all_addresses = np.zeros(self.on_screen, dtype = int)
		
		self.handle = handle
		self.on = False

		self.start_time = time.time()
		
	def decode_events(self, data_byte):
		time_stamps = []
		addresses = []
		data = bitstring.BitArray(data_byte).bin
		event_nr = int(len(data)/64)
	
		if event_nr > 0:
			for e in range(event_nr):
				event = data[e*64:e*64+64]
				
				if int(event[24:54], 2) < self.last_event or int(event[54:], 2) == 0:
					# print(data[e*64:e*64+64]) # prints out incorrectly read events; uncomment this line to see how many errors event read-out circuit makes
					None
				else: 
					time_stamps.append(1e-6*int(event[24:54], 2))
					# time_stamps.append(time.time()-self.start_time)		## time stamping on PC (NOT RECOMMENDED)
					addresses.append(int(event[54:], 2))       
				self.last_event = int(event[24:54], 2)

		return time_stamps, addresses

	def ReadEvents(self):
		try:
			event_data = self.handle.bulkRead(endpoint=0x81, length=1024*16, timeout=1)
			time_stamps, addresses = self.decode_events(event_data)
	
			dn = len(time_stamps)
			if dn > 0:
				self.all_time_stamps = np.roll(self.all_time_stamps, -dn)
				self.all_addresses = np.roll(self.all_addresses, -dn)
				self.all_time_stamps[-dn:] = np.array(time_stamps)
				self.all_addresses[-dn:] = np.array(addresses)
				
				self.spikes_curve.setData(x=self.all_time_stamps, y=self.all_addresses, _callSync='off')
	
		except usb1.USBErrorTimeout:
			# time.sleep(100e-3)
			None
		except usb1.USBErrorIO:
			None
			# print('USBErrorIO during ReadEvents')

#############################################################################################################

class handle_weights(object):
	def __init__(self, handle):
		self.S = 8192
		# self.x = np.loadtxt('config/x.txt')
		
		self.TERM = binascii.unhexlify('FFFFFFFF')
		self.SP_RON = binascii.unhexlify('F5000000')
		self.RWT = binascii.unhexlify('F9000000')
		self.ECHO = binascii.unhexlify('FF00AF00')
		
		self.weights = np.zeros(self.S, dtype=int)
		self.post = 10 # number of postsynaptic neurons
		self.post_list = [450, 460, 470, 480, 490, 500, 510, 520, 530, 540]
		self.im_width = 20
		self.n_tot = self.im_width**2
		
		# Plotting color scheme definition
		self.pos = np.array([0.0, 0.5, 1.0])
		self.color = np.array([[29,82,135,255], [53,193,133,255], [208,246,106,255]])
		self.color_map = pg.ColorMap(self.pos, self.color, mode=None)

		self.app = pg.mkQApp()
		# self.proc = mp.QtProcess()
		# self.rpg = self.proc._import('pyqtgraph')

		self.win = QtGui.QMainWindow()
		self.win.resize(1600,400)
		self.cw = QtGui.QWidget() # central widget
		self.win.setCentralWidget(self.cw)
		self.l = QtGui.QGridLayout() # layout
		self.cw.setLayout(self.l)

		self.subplots = []
		for i in range(self.post):
			# self.subplots.append(pg.image(np.zeros((self.im_width, self.im_width))))
			self.subplots.append(pg.ImageView())
			# self.subplots[i].view.setAspectLocked(False)
			self.subplots[i].setColorMap(self.color_map)
			self.subplots[i].resize(340,170)
			self.subplots[i].ui.roiBtn.hide()
			self.subplots[i].ui.menuBtn.hide()

			self.l.addWidget(self.subplots[i], int(i/5), i-int(i/5)*5)

		self.win.show()

		self.handle = handle
		
	def from_bytes(self, data_byte=bytearray()): # takes binary data and converts it to decimal values
		bin_str = bitstring.BitArray(data_byte).bin
		# bin_str = bin_str[:524288]
		if len(bin_str) != self.S*64:
			print('ERROR: Provided binary array size (' + str(len(bin_str)) + ' bits) does not match the expected size of ' + str(self.S*64) + ' bits.')
			sys.exit()

		w_dec = np.zeros(self.S, dtype=int)
		for i in range(int(len(bin_str)/64)):
		   w_dec[i] = int(bin_str[i*64+30:i*64+41], 2)

		return w_dec
	
	def read_data(self):
		tout = 0
		byte_data = bytearray()
		
		handle.bulkWrite(endpoint=0x01, data=self.TERM, timeout=1)
		handle.bulkWrite(endpoint=0x01, data=self.ECHO, timeout=1)
		while True:
			if tout == 3:
				tout = 0
				break
			else:
				try:
					handle.bulkRead(endpoint=0x81, length=16*1024, timeout=1)
					tout = 0
				except usb1.USBErrorTimeout:
					time.sleep(1e-3)
					tout += 1
				except usb1.USBErrorIO:
					# print('USBErrorIO during bulkRead')
					tout += 1

		self.handle.bulkWrite(endpoint=0x01, data=self.RWT, timeout=1)
		while True:
			if tout == 3:
				self.handle.bulkWrite(endpoint=0x01, data=self.SP_RON, timeout=1)
				break
			else:
				try:
					byte_data += self.handle.bulkRead(endpoint=0x81, length=1024*2, timeout=1)
					tout = 0
				except usb1.USBErrorTimeout:
					time.sleep(1e-3)
					tout += 1
		return byte_data
	
	def plot_data(self, byte_data):
		weights_rd = self.from_bytes(byte_data)
		# weights_rd = np.array(list(zip(*sorted(zip(self.x, weights_rd))))[1])
		weights = []
		
		for i in range(self.post):
			weights.append(np.zeros((self.im_width, self.im_width), dtype=int))
		
		for row in range(self.im_width):
			for col in range(self.im_width):
			   for i in range(self.post):
				   weights[i][row,col] = weights_rd[self.post*(self.im_width*row+col)+i]
		
		for i in range(self.post):
			weights[i] = weights[i]/(2**11-1)
		
		for i in range(self.post):
			self.subplots[i].setImage(np.transpose(weights[i]), autoRange=True)
			pg.QtGui.QApplication.processEvents()
		
#############################################################################################################

def clean(handle):
	tout = 0
	handle.bulkWrite(endpoint=0x01, data=binascii.unhexlify('FFFFFFFF'), timeout=1) ## TERM
	handle.bulkWrite(endpoint=0x01, data=binascii.unhexlify('FFFFFF00'), timeout=1) ## ECHO
	while True:
		if tout == 10:
			tout = 0
			break
		else:
			try:
				handle.bulkRead(endpoint=0x81, length=2*1024, timeout=1)
				tout = 0
			except usb1.USBErrorTimeout:
				time.sleep(1e-3)
				tout += 1

#############################################################################################################                
#QtGui.QApplication.instance().exec_()
FX3_VID = 0x04B4
FX3_PID = 0x00F1

T_events = 2 # Time (in seconds) for applkying spikes before reading weights

		
def run_plot():
	global EventPlotter, spikes_on
	while spikes_on == True:
		EventPlotter.ReadEvents()
		
# def run_bar():
#     global ImageSpikeGenerator, spikes_on
#     while spikes_on == True:
#         ImageSpikeGenerator.NextFrame()
		
def cmd_input():
	global spikes_on, skip, EventPlotter
	while True:
		cmd = input('Type "stop" to stop the cycle ("clear" to clear graph)...\n')
		if cmd == 'stop':
			spikes_on = False
			skip = True
			break
			time.sleep(100e-3)
		if cmd == 'clear':
			EventPlotter.all_time_stamps = np.zeros(EventPlotter.on_screen)
			EventPlotter.all_addresses = np.zeros(EventPlotter.on_screen, dtype = int)
			EventPlotter.last_event = 0

with usb1.USBContext() as context:
	handle = context.openByVendorIDAndProductID(FX3_VID, FX3_PID, skip_on_error=True)
	print('Handle:', handle)
	
	with handle.claimInterface(0):
		clean(handle)

		synapses = rw_lib.synapses()   
		ImageSpikeGenerator = ImageSpikeGenerator()
		usbEventSender = usbEventSender(handle=handle)
		EventPlotter = EventPlotter(handle=handle)
		handle_weights = handle_weights(handle=handle)

		while True:
			try:
				spikes_on = False # variable for stopping system stimulation during weight read-out
				skip = False # variable for skipping/stopping event system excitation and read-out
				cmd = input('Press Enter for the selected image generation, or enter a path to an image...\n')

				### Choosing image generation functon between rotating bar and image loading
				if cmd == '':
					ImageSpikeGenerator.max_activity = 100 # <------------ Change FIRING RATE for learning here <------------
					ImageSpikeGenerator.hop = 30 # 30 for traveling particle; 200 for bar
					ImageSpikeGenerator.count = ImageSpikeGenerator.hop
					ImageSpikeGenerator.load_wav(path='song.wav', t_disp=50e-3)
					ImageSpikeGenerator.load_DVS()
					def rot_func():
						# spike_matrix = ImageSpikeGenerator.NextFrame(func='draw_wave')
						# spike_matrix = ImageSpikeGenerator.NextFrame(func='draw_particle')
						# spike_matrix = ImageSpikeGenerator.NextFrame(func='draw_bar')
						spike_matrix = ImageSpikeGenerator.draw_DVS()
						usbEventSender.SendEvents(spike_matrix=spike_matrix)
					### Launching a thread that will wait for user commands
					thr_cmd = threading.Thread(target=cmd_input)
					thr_cmd.daemon = False
					thr_cmd.start()	

				elif cmd == 'quit':
					skip = True
					spikes_on = False
					time.sleep(10e-3)
					EventPlotter.proc.close()
					sys.exit()
				else:
					try:
						file = open(cmd, 'r')
					except FileNotFoundError:
						print('File not found. Please check the file name.')
						skip = True
					if skip == False:
						ImageSpikeGenerator.max_activity = 2 # <------------ Change FIRING RATE for inference here <------------
						def rot_func():
							spike_matrix = ImageSpikeGenerator.NextFrame(func='draw_img', img_path=cmd)
							usbEventSender.SendEvents(spike_matrix=spike_matrix)
						### Launching a thread that will wait for user commands
						thr_cmd = threading.Thread(target=cmd_input)
						thr_cmd.daemon = False
						thr_cmd.start()	
				### Done choosing a function

				# ### Launching a thread that will wait for user commands
				# thr_cmd = threading.Thread(target=cmd_input)
				# thr_cmd.daemon = False
				# thr_cmd.start()	

				while skip == False:
					try:
						time_mark = time.time()

						### Launching a thread for event plotting
						thr_plot = threading.Thread(target=run_plot)
						thr_plot.daemon = True
						thr_plot.start()

						while spikes_on == True:
							rot_func()
							# time.sleep(1e-3)
							if time.time()-time_mark > T_events: # stop event writing after T_events time period
								spikes_on = False

						### Reading out weights
						weights = handle_weights.read_data()
						handle_weights.plot_data(weights)
						spikes_on = True

					except KeyboardInterrupt:
						break
			except KeyboardInterrupt:
				break

EventPlotter.proc.close()
sys.exit()

		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		