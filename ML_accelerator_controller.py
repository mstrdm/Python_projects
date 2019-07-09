## This script is used for controlling the FPGA based ML accelerator through GPIO from Raspberry Pi 3 Model B board.
## MNIST data is used in this controller script to drive the accelerator.
import numpy as np
import RPi.GPIO as gpio
import time
import struct
import sys
import random

print "**** FPGA MCHL controller v1.2 ****\n"

gpio.setwarnings(False)

test_n = 10000
train_n = 60000
IN_B = 2
tot_bits = IN_B*784 + 4 # number of bits per single image: bits_per_pixel*pixels + binary_label

sipo_size = 16

sipo_in_gpio = [2,3,4,17,27,22,10,9,14,15,18,23,24,25,8,7]
sipo_next_gpio = [11]
cmd_rd_gpio = [5]
piso_wr_gpio = [6]
piso_next_gpio = [13]
piso_out_gpio = [12,16,20,21]

unused_gpio = [19,26]

gpio_out = sipo_in_gpio + sipo_next_gpio + cmd_rd_gpio + piso_wr_gpio + piso_next_gpio + unused_gpio
gpio_in = piso_out_gpio

gpio.setmode(gpio.BCM)
for i in gpio_out:
	gpio.setup(i, gpio.OUT)
	gpio.output(i, 0)
for i in gpio_in:
	gpio.setup(i, gpio.IN)


def gpio_set(port, val):
	while gpio.input(port) != val:
		gpio.output(port,val)


def gpio_pulse(port):
	while gpio.input(port) != 1:
		gpio.output(port,1)
	while gpio.input(port) != 0:
		gpio.output(port,0)


def load_data(file_name = "test_data.txt"):
	print "[DATA] Loading " + file_name + " data..."

	t0 = time.time()
	data = np.loadtxt(file_name, dtype=int)

	print "[DATA] Data loaded. Took " + str(time.time()-t0) + " seconds."
	return data


def load_data_bin(file_name = "test_data_bin.txt", n = 10000, bits = tot_bits):
	print "[DATA] Loading " + file_name + " data..."

	t0 = time.time()
	
	newFile = open(file_name, "rb")
	raw_data = newFile.read()
	data_size = len(raw_data)
	byte_data = struct.unpack('<' + 'B'*data_size, raw_data)

	bit_data_str = ''
	for i in xrange(data_size):
	    bit_data_str += ('{0:08b}').format(byte_data[i])

	bit_data = map(int, list(bit_data_str))

	data = np.reshape(bit_data, [bits,n])	

	print "[DATA] Data loaded. Took " + str(time.time()-t0) + " seconds."
	return data


def load_data_orig(n = 10000, train = False, bits = tot_bits):
	print "Loading data..."
	t0 = time.time()

	mnist_data = MNISTexample(0, n, train)
	data_dec = np.zeros((784, n), dtype = int)
	label_dec = np.zeros(n, dtype = int)

	for i in xrange(n):
		data_dec[:,i] = np.array(3.9*np.array(mnist_data[i][0]), dtype = int)
		label_dec[i] = mnist_data[i][1].index(1)

	data = np.zeros((bits, n), dtype = int)

	for i in xrange(n):
		data_bin = ''
		data_bin += ('{0:04b}').format(label_dec[i])

		for ii in xrange(784):
			data_bin += ('{0:02b}').format(data_dec[ii,i])
		data[:,i] = map(int, list(data_bin))

	print "Data loaded. Took " + str(time.time()-t0) + " seconds."
	return data


def inference(test_data, test_n = 1, sipo_size = sipo_size, bits = IN_B*784, layer = 0):
	# command: learn, layer[1:0], label[3:0], infer_reset, zeroes
	
	layer_bin = map(int, list(('{0:02b}').format(layer)))

	for i in xrange(test_n):

		data = np.zeros(sipo_size, dtype = int)
		data[1:3] = layer_bin # layer[1:0]
		data[3:7] = test_data[0:4,i]

		if i == 0:
			data[7] = 1 # refresh output counter



		gpio.output(sipo_in_gpio[15], data[15])
		gpio.output(sipo_in_gpio[14], data[14])
		gpio.output(sipo_in_gpio[13], data[13])
		gpio.output(sipo_in_gpio[12], data[12])
		gpio.output(sipo_in_gpio[11], data[11])
		gpio.output(sipo_in_gpio[10], data[10])
		gpio.output(sipo_in_gpio[9], data[9])
		gpio.output(sipo_in_gpio[8], data[8])
		gpio.output(sipo_in_gpio[7], data[7])
		gpio.output(sipo_in_gpio[6], data[6])
		gpio.output(sipo_in_gpio[5], data[5])
		gpio.output(sipo_in_gpio[4], data[4])
		gpio.output(sipo_in_gpio[3], data[3])
		gpio.output(sipo_in_gpio[2], data[2])
		gpio.output(sipo_in_gpio[1], data[1])
		gpio.output(sipo_in_gpio[0], data[0])

		gpio_pulse(sipo_next_gpio[0])

		# gpio.output(sipo_next_gpio, 1)
		# gpio.output(sipo_next_gpio, 0)		

		for k in xrange(int(bits/sipo_size)):

			data = test_data[4+sipo_size*k:4+sipo_size*(k+1),i]

			gpio.output(sipo_in_gpio[15], data[15])
			gpio.output(sipo_in_gpio[14], data[14])
			gpio.output(sipo_in_gpio[13], data[13])
			gpio.output(sipo_in_gpio[12], data[12])
			gpio.output(sipo_in_gpio[11], data[11])
			gpio.output(sipo_in_gpio[10], data[10])
			gpio.output(sipo_in_gpio[9], data[9])
			gpio.output(sipo_in_gpio[8], data[8])
			gpio.output(sipo_in_gpio[7], data[7])
			gpio.output(sipo_in_gpio[6], data[6])
			gpio.output(sipo_in_gpio[5], data[5])
			gpio.output(sipo_in_gpio[4], data[4])
			gpio.output(sipo_in_gpio[3], data[3])
			gpio.output(sipo_in_gpio[2], data[2])
			gpio.output(sipo_in_gpio[1], data[1])
			gpio.output(sipo_in_gpio[0], data[0])

			gpio_pulse(sipo_next_gpio[0])

			# gpio.output(sipo_next_gpio, 1)
			# gpio.output(sipo_next_gpio, 0)

		gpio_pulse(cmd_rd_gpio[0])

		# gpio.output(cmd_rd_gpio, 1)
		# gpio.output(cmd_rd_gpio, 0)

	out_data = np.zeros(16, dtype = int)

	gpio_pulse(piso_wr_gpio[0])

	# gpio.output(piso_wr_gpio, 1)
	# gpio.output(piso_wr_gpio, 0)

	for x in xrange(4):
		out_data[4*x + 0] = int(gpio.input(piso_out_gpio[0]))
		out_data[4*x + 1] = int(gpio.input(piso_out_gpio[1]))
		out_data[4*x + 2] = int(gpio.input(piso_out_gpio[2]))
		out_data[4*x + 3] = int(gpio.input(piso_out_gpio[3]))

		gpio_pulse(piso_next_gpio[0])

		# gpio.output(piso_next_gpio, 1)
		# gpio.output(piso_next_gpio, 0)

		out_dec = ''.join(str(int(k)) for k in out_data)

	return int(out_dec, 2)

def training(train_data, train_n = train_n, sipo_size = sipo_size, bits = IN_B*784, layer = 0):
	# command: laern, layer[1:0], label[3:0], infer_reset, zeroes
	
	layer_bin = map(int, list(('{0:02b}').format(layer)))

	for i in xrange(train_n):

		N = int(60000*random.random())

		data = np.zeros(sipo_size, dtype = int)
		data[0] = 1 # learning
		data[1:3] = layer_bin # layer[1:0]
		data[3:7] = train_data[0:4,N] # label

		gpio.output(sipo_in_gpio[15], data[15])
		gpio.output(sipo_in_gpio[14], data[14])
		gpio.output(sipo_in_gpio[13], data[13])
		gpio.output(sipo_in_gpio[12], data[12])
		gpio.output(sipo_in_gpio[11], data[11])
		gpio.output(sipo_in_gpio[10], data[10])
		gpio.output(sipo_in_gpio[9], data[9])
		gpio.output(sipo_in_gpio[8], data[8])
		gpio.output(sipo_in_gpio[7], data[7])
		gpio.output(sipo_in_gpio[6], data[6])
		gpio.output(sipo_in_gpio[5], data[5])
		gpio.output(sipo_in_gpio[4], data[4])
		gpio.output(sipo_in_gpio[3], data[3])
		gpio.output(sipo_in_gpio[2], data[2])
		gpio.output(sipo_in_gpio[1], data[1])
		gpio.output(sipo_in_gpio[0], data[0])

		gpio_pulse(sipo_next_gpio[0])

		# gpio.output(sipo_next_gpio, 1)
		# gpio.output(sipo_next_gpio, 0)

		for k in xrange(int(bits/sipo_size)):

			data = train_data[4+sipo_size*k:4+sipo_size*(k+1),N]

			gpio.output(sipo_in_gpio[15], data[15])
			gpio.output(sipo_in_gpio[14], data[14])
			gpio.output(sipo_in_gpio[13], data[13])
			gpio.output(sipo_in_gpio[12], data[12])
			gpio.output(sipo_in_gpio[11], data[11])
			gpio.output(sipo_in_gpio[10], data[10])
			gpio.output(sipo_in_gpio[9], data[9])
			gpio.output(sipo_in_gpio[8], data[8])
			gpio.output(sipo_in_gpio[7], data[7])
			gpio.output(sipo_in_gpio[6], data[6])
			gpio.output(sipo_in_gpio[5], data[5])
			gpio.output(sipo_in_gpio[4], data[4])
			gpio.output(sipo_in_gpio[3], data[3])
			gpio.output(sipo_in_gpio[2], data[2])
			gpio.output(sipo_in_gpio[1], data[1])
			gpio.output(sipo_in_gpio[0], data[0])

			gpio_pulse(sipo_next_gpio[0])

			# gpio.output(sipo_next_gpio, 1)
			# gpio.output(sipo_next_gpio, 0)

		gpio_pulse(cmd_rd_gpio[0])

		# gpio.output(cmd_rd_gpio, 1)
		# gpio.output(cmd_rd_gpio, 0)
		# time.sleep(1e-3)


def cont_train(train_data, test_data, train_n = 500e3, test_n = 10000, sipo_size = sipo_size, bits = IN_B*784, points = 50, layer = 0):
	runs = int(train_n/points) # here points is the number of points per accuracy curve
	result = np.zeros(points, dtype=int)

	# print "[TRAINING] Training Layer " + str(layer) + "..."
	t0 = time.time()
	# training(train_data=train_data, train_n=int(runs), layer=0)

	# print "[TRAINING] Done training Layer " + str(layer) + ". Training took " + str((time.time()-t0)/3600.) + " hours."

	print "[TRAINING] Training Layer " + str(layer) + "..."

	for i in xrange(points):
		training(train_data=train_data, train_n=runs, layer=layer) # Change layer to 1 if training two layers!!!
		result[i] = inference(test_data=test_data, test_n=int(test_n), layer=layer)
		print "[INFERENCE] Epoch nr.: " + str(i) + "; Accuracy: " + str(result[i])

	print "[TRAINING] Done training Layer " + str(layer) + ". Training took " + str((time.time()-t0)/3600.) + " hours."

	return result

############################################################################################################
######################################### Beginning of the script ##########################################
############################################################################################################

test_data = load_data_bin("test_data_bin.txt")

# while True:
t0 = time.time()
A = inference(test_data=test_data, test_n=test_n)
for i in gpio_out:
	gpio.output(i, 0)
print "[INFERENCE] Inference took " + str(time.time()-t0) + " seconds. Accuracy: " + str(A)


train_data = load_data("train_data.txt")

l0_result = cont_train(train_data=train_data, test_data=test_data, test_n=test_n, layer=0)
np.savetxt("l0_result.txt", l0_result)

l1_result = cont_train(train_data=train_data, test_data=test_data, test_n=test_n, layer=1)
np.savetxt("l1_result.txt", l1_result)

# t0 = time.time()
# training(train_data=train_data, train_n=10000, layer=0)
# training(train_data=train_data, train_n=10000, layer=1)
# A = inference(test_data=test_data, test_n=test_n)
# print time.time()-t0, A

# while True:
# 	cont = raw_input("One more run? Type 'y' to continue: ")
# 	if cont == "y":
# 		t0 = time.time()
# 		training(train_data=train_data, train_n=10000, layer=0)
# 		training(train_data=train_data, train_n=10000, layer=1)
# 		A = inference(test_data=test_data, test_n=test_n)
# 		print time.time()-t0, A

# 	else: sys.exit("Exiting...")



