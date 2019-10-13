import numpy as np
class Conv2d(object):
	def __init__(self, in_channels, out_channels, kernel_size
	, stride = 1, padding = 0,bias = False):
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		k = in_channels*kernel_size[0]*kernel_size[1]
		self.kernel = np.random.uniform(-np.sqrt(k),np.sqrt(k),(out_channels,in_channels,kernel_size[0],kernel_size[1]))
		self.is_bias = bias
#		if bias:
#			self.bias = np.random.uniform(-np.sqrt(k),np.sqrt(k),())
	def forward(self,x):
		input_shape = x.shape
		padded_height  = input_shape[2]+self.padding[0]*2
		padded_width  = input_shape[3]+self.padding[1]*2
		padded_x = np.zeros((input_shape[0],input_shape[1],padded_height,padded_width))
		padded_x[:,:,self.padding[0]:-self.padding[0],self.padding[1]:-self.padding[1]] = x
		self.reg = padded_x
		#print(padded_x.shape)
		height = (padded_height - self.kernel_size[0] // self.stride[0]) + 1
		width  = (padded_width  - self.kernel_size[1] // self.stride[1]) + 1
		batch_size = input_shape[0]
		out_x = np.zeros((batch_size, self.out_channels, height, width))
		x = 0
		y = 0
		for image in range(batch_size):
			x = 0
			for i in range(0,height,self.stride[0]):
				y = 0
				for j in range(0,width,self.stride[1]):
					out_x[image,:,x,y] = (self.kernel * padded_x[image,:,i:i+self.kernel_size[0],j:j+self.kernel_size[1]]).sum(axis=(1,2,3))
					y += 1
				x += 1
		return out_x
	def backprop(self, delta_loss, lr):
		delta_kernel = np.zeros(self.kernel.shape)
		batch_size = self.reg.shape[0]
		x = 0
		y = 0
		for image in range(batch_size):
			x = 0
			for i in range(0,delta_loss.shape[2],self.stride[0]):
				y = 0
				for j in range(0,delta_loss.shape[3],self.stride[1]):
					delta_kernel += (delta_loss[image,:,x,y].reshape(-1,1,1,1) * np.broadcast_to(self.reg[image,:,i:i+self.kernel_size[0],j:j+self.kernel_size[1]],self.kernel.shape))
					y += 1
				x += 1
		delta_kernel /= batch_size
		origin_kernel = self.kernel
		self.kernel -= lr * delta_kernel
		if self.is_bias:
			self.bias -= lr * delta_loss
		#calculate convolution of delta_loss(padded) and kernel(flipped) means the gradient of input
		flipped_kernel = np.flip(origin_kernel,axis=(2,3)).transpose(1,0,2,3)
		padded_loss = np.zeros((delta_loss.shape[0],delta_loss.shape[1],delta_loss.shape[2]+2,delta_loss.shape[3]+2))
		padded_loss[:,:,1:-1,1:-1] = delta_loss
		delta_x = np.zeros(self.reg.shape)
		for image in range(batch_size):
			x = 0
			for i in range(0,self.reg.shape[2],self.stride[0]):
				y = 0
				for j in range(0,self.reg.shape[3],self.stride[1]):
					delta_x[image,:,x,y] = (flipped_kernel * padded_loss[image, : , i:i+self.kernel_size[0], j:j+self.kernel_size[1]]).sum(axis=(1,2,3))
					y += 1
				x += 1
		delta_x = delta_x[:,:,self.padding[0]:-self.padding[0],self.padding[1]:-self.padding[1]]
		return delta_x
