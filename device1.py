### import library for arduino 
#import serial
#from serial.tools import list_ports

### import library for analog discovery device
from analog_discovery import AnalogDiscovery

from dsp_utils import DSPUtils
import numpy as np  

class Device1:

	SAMPLE_DEVICE_ARDUINO = 0
	SAMPLE_DEVICE_ANALOG_DISCOVERY = 1

	SAMPLE_RATE = 1024
	BUFFER_SIZE = 512
	SHIFT_SIZE = 512
	# DOWNSAMPLE_RATIO = 1/100

	def __init__(self, option):
		self.option = option
		self.background_fft_profile = None
		self.internal_device = None
		if self.option== Device1.SAMPLE_DEVICE_ANALOG_DISCOVERY:
			print('connecting to first xANALOG DISCOVERY device')
			self.internal_device = AnalogDiscovery(Device1.SAMPLE_RATE, Device1.BUFFER_SIZE, Device1.SHIFT_SIZE,1)
			self.internal_device.open_analog_discovery()

	def calculate_background_fft_profile(self):

		self.background_fft_profile = []
		for _ in range (10):
			sig = self.sample()
			while len(sig) <= 0:
				sig = self.sample()
			filtered_signal = sig
			filtered_signal = filtered_signal
		    # filtered_signal = apply_window_filter(filtered_signal)
			fft = DSPUtils.calculate_fft(filtered_signal, Device1.BUFFER_SIZE)

			if len(self.background_fft_profile) > 0:
				self.background_fft_profile = np.amax([fft, self.background_fft_profile], axis = 0)
			else:
				self.background_fft_profile = fft

		return self.background_fft_profile
        # print(len(fft))
        # print(fft)
        # fft = np.flip(fft)

	def sample_from_analog_discovery(self):
		if len(self.internal_device.in_buffer) >0 :
			data= self.internal_device.in_buffer.pop(0)
			signal = np.array(data)
			return signal
		return np.array([])	


	def sample(self):
	    if self.option == Device1.SAMPLE_DEVICE_ANALOG_DISCOVERY:
	        return self.sample_from_analog_discovery()




