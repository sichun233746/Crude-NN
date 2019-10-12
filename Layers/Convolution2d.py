import numpy as np
class Conv2d(object):
	def __init__(self, in_channels, out_channels, kernel_size
	, stride = 1, padding = 0,bias = True):
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.kernel = np.zeros((out_channels,in_channels,kernel_size[0],kernel_size[1]))
	def forward(self,x):
		input_shape = x.shape
		padded_height  = input_shape[2]+self.padding[0]*2
		padded_width  = input_shape[3]+self.padding[1]*2
		padded_x = np.zeros((input_shape[0],input_shape[1],padded_height,padded_width))
		for i in range(input_shape[0]):
			for j in range(input_shape[1]):
				padded_x[i,j,self.padding[0]:-self.padding[0],self.padding[1]:-self.padding[1]] = x[i][j]
		height = (padded_height - self.kernel_size[0] // self.stride[0]) + 1
		width  = (padded_width  - self.kernel_size[1] // self.stride[1]) + 1
		batch_size = input_shape[0]
		out_x = np.zeros((batch_size, self.out_channels, height, width))
		x = 0
		y = 0
		for image in range(batch_size):
			y = 0
			for i in range(0,input_shape[2],self.stride[0]):
				x = 0
				for j in range(0,input_shape[3],self.stride[1]):
					out_x[image,:,x,y] = (self.kernel * padded_x[image,:,i:i+self.stride[0],j:j+self.stride[1]]).sum(axis=(1,2,3))
					x += 1
				y += 1
		print(out_x.shape)

