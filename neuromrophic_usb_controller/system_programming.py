## Basic script for programming neuromorphic chip through FX3 USB3 controller

import numpy as np
import usb1
import time
import binascii
import bitstring
import sys
sys.path.append('lib')
import rw_lib

FX3_VID = 0x04B4
FX3_PID = 0x00F1

TERM =      'FFFFFFFF'
WW =        'F1000000'
WT =        'F2000000'
WP =        'F3000000'
SP_ROFF =   'F4000000'
SP_RON =    'F5000000'
RWT =       'F9000000'
RP =        'FB000000'
ECHO =      'FF00AF00' # ECHO will return sent value: 0xFFxxxxx0

synapses = rw_lib.synapses()
parameters = rw_lib.parameters()
parameters.load('config/param.ini')
param_byte = parameters.to_bytes(default=False)

with open('config/topology_wr_bin', 'rb') as f:
    top_byte = f.read()
with open('config/weights_wr_bin', 'rb') as f:
    w_byte = f.read()

byte_data = bytearray()

with usb1.USBContext() as context:
    handle = context.openByVendorIDAndProductID(FX3_VID, FX3_PID, skip_on_error=True)
    print(handle)

    if handle is None:
        print('Device not present, or user is not allowed to access device.')

    with handle.claimInterface(0):
#        usb_data = binascii.unhexlify(WT) + top_byte
#        usb_data = binascii.unhexlify(WW) + w_byte
#        usb_data = binascii.unhexlify(RWT)
#        usb_data = binascii.unhexlify(WP) + param_byte
#        usb_data = binascii.unhexlify(WP) + param_byte + binascii.unhexlify(RP)
#        usb_data = binascii.unhexlify(RP)
#        usb_data = binascii.unhexlify(ECHO)
        usb_data = binascii.unhexlify(TERM)
#        usb_data = binascii.unhexlify(SP_RON)
        handle.bulkWrite(endpoint=0x01, data=usb_data, timeout=1)

        tout = 0
        try:
            while True:
                if tout == 10:
                    print('Read', len(byte_data), 'bytes')
                    data_out = bitstring.BitArray(byte_data).bin
                    f = open('config/data_rd_bin', 'wb')
                    f.write(byte_data)
                    f.close()
                    break
                else:
                    try:
                        byte_data += handle.bulkRead(endpoint=0x81, length=1024*2, timeout=1)
                        tout = 0
                    except usb1.USBErrorTimeout:
                        time.sleep(1e-3)
                        tout += 1
        except KeyboardInterrupt:
            print('ByeBye')
































