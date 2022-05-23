
import numpy as np

def relu(data):
	return max(0, data)


class Layer():
	layer_count = 0
	def __init__(self, neuron_number, input_number):
		self.neuron_number = neuron_number
		self.input_number = input_number

		self.layer_index = Layer.layer_count
		Layer.layer_count += 1

		self.input_X = np.zeros( (self.input_number,) )
		self.Z = np.zeros( (self.neuron_number) )

		self.weights = np.zeros( (self.neuron_number, self.input_number) )
		self.bias = np.zeros( (self.neuron_number,) )

		self.delta = np.zeros( (self.neuron_number, ) )

	def init_layer(self):

		if self.input_number != len(self.input_X):
			print("error !\n")
			return 0

		for n in range(0, self.neuron_number):
			
			for i in range(0, self.input_number):
				self.weights[n][i] = 0.1
			self.bias[n] = 0.1

	def show_info(self):
		print("______________________________________")
		print("Layer n° ", self.layer_index)
		print("bias : ", self.bias)
		print("weights : \n", self.weights)
		print("output data : ", self.Z)
		print("input_X : ", self.input_X)
		print("delta : ", self.delta)
		print("______________________________________\n")


	def forward_layer(self):
		for n in range(0, self.neuron_number):
			self.Z[n] = 0
			z = 0
			for i in range(0, self.input_number):
				z += ( self.weights[n][i]) * (self.input_X[i])
			self.Z[n] = z
			self.Z[n] += self.bias[n]
			self.Z[n] = relu(self.Z[n])

		


class neural_network(Layer):
	layer_count = 0
	def __init__(self, layer_number, X_in, Y_in):
		self.layer_number = layer_number
		self.X_in = X_in[0]
		self.Y_in = Y_in[0]
		self.layers = np.zeros( (layer_number,), Layer )

	def compile_NeuralNetwork(self):
		if self.layer_number != neural_network.layer_count:
			print("error")

		if len(self.X_in) != len(self.layers[0].input_X):
			print("error")
			return 0

		for l in range(0, len(self.layers) ):
			self.layers[l].init_layer()

		self.Z_out = np.zeros( ( len(self.layers[neural_network.layer_count-1].Z), ) )

		self.layers[0].input_X = self.X_in

		for l in range(0, len(self.layers)-1):
			if len(self.layers[l+1].input_X) != len(self.layers[l].Z):
				print("error dimension layers")
				return 0
			self.layers[l+1].input_X = self.layers[l].Z	


	def forward_nn(self):
		for l in range(0, len(self.layers)):
			self.layers[l].forward_layer()
		self.Z_out = self.layers[neural_network.layer_count-1].Z
		print("Z_out : ", self.Z_out)

	def backward_nn(self, learning_rate):
		self.layers[neural_network.layer_count-1].delta = self.Y_in - self.Z_out
		print("delta out : ", self.layers[neural_network.layer_count-1].delta)

		for l in range(len(self.layers)-2, -1, -1):
			#print("layer", l)
			for n in range(0,self.layers[l].neuron_number):
				#print("neuron ", n)
				self.layers[l].delta[n] = self.layers[l-1].weights[0][n]*self.layers[l-1].delta

		for l in range(len(self.layers)-1, -1, -1):
			for n in range(0, self.layers[l].neuron_number):
				for i in range(0, self.layers[l].input_number):
					#print("weights", i, "_", n, "_", l)
					self.layers[l].weights[n][i] = self.layers[l].weights[n][i] - learning_rate*self.layers[l].input_X[i]*self.layers[l].delta[n]
					

				

	def train_nn(self, learning_rate, features, labels ):
		for e in range(0,len(features)):
			self.X_in = features[e]
			self.Y_in = labels[e]

			self.forward_nn()
			self.backward_nn(learning_rate)
		

	def add_layer(self, Layer ):
		neural_network.layer_count += 1
		self.layers[Layer.layer_index] = Layer

	def show_info(self):
		print("nb total de layers : ", self.layer_number)
		for l in range(0, len(self.layers) ):
			print(self.layers[l].show_info())
		print("accuracy : ", int((1-abs(self.layers[neural_network.layer_count-1].delta))*100), "%" )




X = [1, 1]
Y = 0

data = [[1,1],[0,0],[1,0],
		[0,1],[1,1],[0,1],
		[1,1],[1,0], [1,1],
		[0,0], [0,1], [1,1],
		[1,0], [0,0], [0,1]]


labels = [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1 ]

nn = neural_network(layer_number=2, X_in=data, Y_in=labels )

nn.add_layer( Layer(4, 2) )
nn.add_layer( Layer(1, 4) )

nn.compile_NeuralNetwork()

nn.train_nn(0.1, data, labels)

nn.show_info()	
print(data[0])




