import numpy as np
from collections import OrderedDict
import bitstring
import binascii
import sys

class const(object): # object holding all important parameters in one place
	def __init__(self):
		self.clk = 100e6 # Hz
		self.param_len = 218
		self.divider = np.log(8./7)
		self.S = 8192 # number of snapses


class parameters(object): # class holding functions for processing neo2c parameter data
	def __init__(self, const=const):
		self.const = const()
		self.clk = self.const.clk
		self.divider = self.const.divider
		self.param_len = self.const.param_len

		self.defval_dic = OrderedDict() # default parameter values
		self.defval_dic['INDEX'] = 0
		self.defval_dic['REFR'] = 0
		self.defval_dic['REFR_RES'] = 9999
		self.defval_dic['TAU_PSP'] = int(20e-3*self.clk*self.divider)
		self.defval_dic['THR'] = 10000
		self.defval_dic['W_INH'] = 5000
		self.defval_dic['W_EXT'] = 10001
		self.defval_dic['TAU_PRE'] = int(20e-3*self.clk*self.divider)
		self.defval_dic['TAU_POST'] = int(80e-3*self.clk*self.divider)
		self.defval_dic['AP'] = 0
		self.defval_dic['AD'] = 0
		self.defval_dic['AP_MAX'] = 255
		self.defval_dic['AD_MAX'] = 255
		self.defval_dic['W_MAX'] = 2047
		self.defval_dic['TS_RES'] = 99
		self.defval_dic['IN_WAIT'] = 1
		self.defval_dic['RELAX_SEL'] = 2
		
		self.curval_dic = OrderedDict() # current parameter values
		self.curval_dic['INDEX'] = 0
		self.curval_dic['REFR'] = 0
		self.curval_dic['REFR_RES'] = 9999
		self.curval_dic['TAU_PSP'] = 20
		self.curval_dic['THR'] = 10000
		self.curval_dic['W_INH'] = 5000
		self.curval_dic['W_EXT'] = 10001
		self.curval_dic['TAU_PRE'] = 20
		self.curval_dic['TAU_POST'] = 80
		self.curval_dic['AP'] = 0
		self.curval_dic['AD'] = 0
		self.curval_dic['AP_MAX'] = 255
		self.curval_dic['AD_MAX'] = 255
		self.curval_dic['W_MAX'] = 2047
		self.curval_dic['TS_RES'] = 99
		self.curval_dic['IN_WAIT'] = 1
		self.curval_dic['RELAX_SEL'] = 2
		
		self.size_dic = OrderedDict() # bit-wise parameter size in Neo2C
		self.size_dic['INDEX'] = 2
		self.size_dic['REFR'] = 10
		self.size_dic['REFR_RES'] = 17
		self.size_dic['TAU_PSP'] = 25
		self.size_dic['THR'] = 14
		self.size_dic['W_INH'] = 14
		self.size_dic['W_EXT'] = 14
		self.size_dic['TAU_PRE'] = 25
		self.size_dic['TAU_POST'] = 25
		self.size_dic['AP'] = 8
		self.size_dic['AD'] = 8
		self.size_dic['AP_MAX'] = 8
		self.size_dic['AD_MAX'] = 8
		self.size_dic['W_MAX'] = 11
		self.size_dic['TS_RES'] = 17
		self.size_dic['IN_WAIT'] = 10
		self.size_dic['RELAX_SEL'] = 2

		self.rdval_dic = OrderedDict() # dictionary of read values
		self.rdval_dic['INDEX'] = 0
		self.rdval_dic['REFR'] = 0
		self.rdval_dic['REFR_RES'] = 0
		self.rdval_dic['TAU_PSP'] = 0
		self.rdval_dic['THR'] = 0
		self.rdval_dic['W_INH'] = 0
		self.rdval_dic['W_EXT'] = 0
		self.rdval_dic['TAU_PRE'] = 0
		self.rdval_dic['TAU_POST'] = 0
		self.rdval_dic['AP'] = 0
		self.rdval_dic['AD'] = 0
		self.rdval_dic['AP_MAX'] = 0
		self.rdval_dic['AD_MAX'] = 0
		self.rdval_dic['W_MAX'] = 0
		self.rdval_dic['TS_RES'] = 0
		self.rdval_dic['IN_WAIT'] = 0
		self.rdval_dic['RELAX_SEL'] = 0

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

		self.curval_dic['REFR'] = int(self.curval_dic['REFR']/0.1)
		self.curval_dic['TAU_PSP'] = int(self.curval_dic['TAU_PSP']*1e-3*self.clk*self.divider)
		self.curval_dic['TAU_PRE'] = int(self.curval_dic['TAU_PRE']*1e-3*self.clk*self.divider)
		self.curval_dic['TAU_POST'] = int(self.curval_dic['TAU_POST']*1e-3*self.clk*self.divider)


	def to_bytes(self, default=False, bname='param_wr_bin'):
		bin_str = '000000' # 6 zeros are added to make parameter array size equal to 224 bits (32*7)
		
		if default == True:
			for item in self.defval_dic.items():
				bin_str += ('{0:0' + str(self.size_dic[item[0]]) + 'b}').format(item[1])
		else:
			for item in self.curval_dic.items():
				if len(bin(item[1])[2:]) > self.size_dic[item[0]]:
					print('ERROR: Max', item[0], 'value is', 2**self.size_dic[item[0]]-1, 'but the requested value is', item[1])
					sys.exit()
				else:
					bin_str += ('{0:0' + str(self.size_dic[item[0]]) + 'b}').format(item[1])

		p_bytes = binascii.unhexlify(bitstring.BitArray(bin=bin_str).hex)

		with open(bname, 'wb') as bin_file:
			bin_file.write(p_bytes)

		return p_bytes
		
	def from_bytes(self, data_byte):
		bin_str = bitstring.BitArray(data_byte).bin
		param_bin = ''
		# Cleaning parameters
		for i in range(int(len(bin_str)/32)):
			param_bin += bin_str[32*i:32*i+32][10:]
		param_bin = param_bin[2:] # removing first two zeros to get 218-bit parameter array
		
		if len(param_bin) != self.param_len:
			print('ERROR: Incorrect binary parameter array size. Expected 218 bits, but got', len(param_bin), 'bits')
			sys.exit()
		else:
			ptr = 0
			print('---READ PARAMETER VALUES---\n')
			for item in self.size_dic.items():
				temp_bin = ''.join(str(int(k)) for k in param_bin[ptr:ptr + item[1]])
				self.rdval_dic[item[0]] = int(temp_bin, 2)
				ptr += item[1]
				if item[0] == 'REFR':
					print(item[0], '=', self.rdval_dic[item[0]], '~', self.rdval_dic[item[0]]/10, 'ms')
				elif item[0] == 'TAU_PSP' or item[0] == 'TAU_PRE' or item[0] == 'TAU_POST':
					print(item[0], '=', self.rdval_dic[item[0]], '~', round(self.rdval_dic[item[0]]*1e3/(self.clk*self.divider), 2), 'ms')
				else:
					print(item[0], '=', self.rdval_dic[item[0]])


class synapses(object): # class for processing neo2c topology and synaptic weight data
	def __init__(self, const=const):
		const.S = 8192
		self.path = '../config/'

	def top_to_bytes(self, fname='topology_wr_dec.txt', bname='topology_wr_bin'):
		with open(self.path+fname, 'r') as topology_file:
			num_lines = sum(1 for line in topology_file)
			
			if num_lines != const.S:
				print('ERROR: Number of rows in', self.path+fname, '(' + str(num_lines) + ') is not equal to the number of synapses in the system (' + str(const.S) + ')')
				sys.exit()
			
			top_bin = ''
			
			topology_file.seek(0)
			for line in topology_file:
				top_line = line.split(',')
				pre_adr = ('{0:010b}').format(int(top_line[0]))
				post_adr = ('{0:010b}').format(int(top_line[1]))
				top_bin += '00000000000' + pre_adr + post_adr + str(int(top_line[2]))
			
			top_bytes = binascii.unhexlify(bitstring.BitArray(bin=top_bin).hex)
					
		with open(self.path+bname, 'wb') as bin_file:
			bin_file.write(top_bytes)

	def w_to_bytes(self, fname='weights_wr_dec.txt', bname='weights_wr_bin'): # takes decimal arrays and conerts them to binary (byte arrays)
		with open(self.path+fname, 'r') as weight_file:
			num_lines = sum(1 for line in weight_file)
			if num_lines != const.S:
				print('ERROR: Number of rows in', self.path+fname, '(' + str(num_lines) + ') is not equal to the number of synapses in the system (' + str(const.S) + ')')
				sys.exit()
			
			w_bin = ''
			weight_file.seek(0)
			for line in weight_file:
				w_bin += '00000' + ('{0:011b}').format(int(line))
		
		w_bytes = binascii.unhexlify(bitstring.BitArray(bin=w_bin).hex)
		with open(self.path+bname, 'wb') as bin_file:
			bin_file.write(w_bytes)
		
	def from_bytes(self, data_byte=bytearray()): # takes binary data and converts it to decimal values
		bin_str = bitstring.BitArray(data_byte).bin
		if len(bin_str) != const.S*64:
			print('ERROR: Provided binary array size (' + str(len(bin_str)) + ' bits) does not match the expected size of ' + str(const.S*64) + ' bits.')
			sys.exit()

		w_dec = np.zeros(const.S, dtype=int)
		top_dec = np.zeros((const.S, 3), dtype=int)
		
		for i in range(int(len(bin_str)/64)):
		   w_dec[i] = int(bin_str[i*64+30:i*64+41], 2)
		   top_dec[i,0] = int(bin_str[i*64+43:i*64+53], 2)
		   top_dec[i,1] = int(bin_str[i*64+53:i*64+63], 2)
		   top_dec[i,2] = int(bin_str[63])

		np.savetxt('weights_rd_dec.txt', w_dec, delimiter=',', fmt='%d')
		np.savetxt('topology_rd_dec.txt', top_dec, delimiter=',', fmt='%d')

#test = parameters()
#test.load('param.ini')
		
	
#test = synapses()
#test.top_to_bytes()
#test.w_to_bytes()

# with open('data_rd_bin', 'rb') as f:
#     a = f.read()
   
# a = test.from_bytes(data_byte=a)



			
			
			
			
