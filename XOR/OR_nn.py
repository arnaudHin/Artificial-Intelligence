

from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy import array, newaxis, random, empty, block, dot, \
	concatenate, full, hstack, exp, linspace, squeeze, \
	log, sum, append


def generate_OR_data(nb_samples):
	X = empty( shape=(nb_samples,2) ) #(m, 2)
	y = empty( shape=(nb_samples,1), dtype=int ) #(m, 1)

	for i in range( int(nb_samples/4) ):
		index = i*4
		X[index] = array([1, 1]) + 0.1* random.randn(1,2)
		y[index] = 1
		X[index+1] = array([1, 0]) + 0.1* random.randn(1,2)
		y[index+1] = 1
		X[index+2] = array([0, 1]) + 0.1* random.randn(1,2)
		y[index+2] = 1
		X[index+3] = array([0, 0]) + 0.1* random.randn(1,2)
		y[index+3] = 0

	dataset = block( [X, y] )
	random.shuffle(dataset)
	return dataset
	
def scatter_dataset( dt ):
	colors = ['red', 'green']
	plt.figure()
	for x1, x2, y in (dt):
		plt.scatter(x1, x2, c=colors[ int(y) ] )
	plt.show()


def plot_cost(J):
	ep = linspace(start=1, stop=len(J), num=len(J))
	plt.figure("COST")
	plt.plot( ep , J, label="J")
	plt.legend()
	plt.show()

def plot_dataset_and_model(dt, w_final, w_evolution):

	if(w_final.ndim > 1): #(neurone, n+1)
		w_final = w_final.squeeze() #(neurone, n+1)

	plt.rcParams.update({'axes.facecolor':'0.5'})
	colors = ['red', 'green']
	plt.figure()
	plt.xlim(-0.5, 1.5)
	plt.ylim(-0.5, 1.5)
	
	for x1, x2, y in (dt):
		plt.scatter(x1, x2, c=colors[ int(y) ] )
	
	for i in range( len(w_evolution) ):
		w_cur = w_evolution[i].squeeze()

		print(w_cur)

		# 0 = x1.w1 + x2.w2 + w0.x0(=1)
		# x1 = a*x2 + b | a = -w1/w2 ; b = -w0/w2
		x = linspace(start=-0.5, stop=1.25, num=dt.shape[0] )
		a, b = -w_cur[1]/w_cur[2], -w_cur[0]/w_cur[2]
		model_trace =  a*x+b

		if( i==0 ):
			plt.plot(x, model_trace, c="blue", lw=3, label="initial")
		elif(i == len(w_evolution)-1):
			plt.plot(x, model_trace, c="red", lw=3, label="final")
		else:
			plt.plot(x, model_trace, c="yellow", lw=2)
	plt.legend()
	plt.show()
	

def sigmoid(z):
    return 1/(1 + exp(-z))

def forward(X_sample, W):

	# W : (neurones, n+1)
	# X_sample : (m, n+1)

	if( W.shape[1] != X_sample.shape[1] ): 
		print("error dim")
		return
	
	z = dot(X_sample, W.T ) #(m, n+1)*(neurones, n+1).T = (m, neurones)
	a = sigmoid(z) #(m, neurones)
	return a


def create_layer(nb_neu, nb_input, activation):
	w = random.randn( nb_neu, nb_input + 1 ) #(neurones, n+1)
	return w


def predict(X_sample, y_sample, W):
	a = forward(X_sample=X_sample, W=W)
	print(f"x_1 : {X_sample[1]} x_2 : {X_sample[2]} | True : {int(y_sample)} " )
	if(a>=0.5):
		print("predict : ", 1)
	else:
		print("predict : ", 0)

"""
log(0) -> impossible
So activation vector must not contain 0
Add an epsilon to assure that A doesn't contain a 0
"""
def compute_loss(A, y):
	#(m, 1)
	epsilon = 1e-15

	j = ( y*log(A+epsilon) + (1-y) * (log(1-A+epsilon)) ) #(m, 1)
	J = (-1/(A.shape[0]) ) * sum(j) #scalar (1,)

	return J


def back_propa(prediction, X_sample, y_sample, actual_W):

	if(X_sample.shape[0] != y_sample.shape[0]): #if #(m, 1) != #(m, 1)
		print("error")
		return

	#cost
	J = compute_loss(A=prediction, y=y_sample) #(1,)

	#delta_computation
	E = prediction-y_sample #(m,1)

	delta = (1/X_sample.shape[0]) * dot(E.T, X_sample) # (m,1).T*(m, n+1) = (1, n+1)
	#batch_gradient_descent
	new_W = actual_W - 0.5*delta # #(neurones, n+1) - 0.1*(neurones, n+1)

	return (new_W, J, E)


#################################################
###                 MAIN PROGRAM              ###
#################################################

m = 100
EPOCHS = 100
DIV = 10

(dataset) = generate_OR_data(nb_samples=m)
#scatter_dataset(dataset)

y = dataset[:,-1] #(m,)
X = dataset[:,:2] #(m, n)
one_col = full(shape=(X.shape[0],1), fill_value=1 ) #(m, 1)
X = hstack( (one_col, X) ) #(m, n+1)
y = y.reshape((y.shape[0], 1)) #(m, 1)

w = create_layer(nb_neu=1, nb_input=2, activation=None)

J = []
w_evo = []

for epoch in tqdm(range(EPOCHS)):

	a = forward(X_sample=X , W=w ) #(m, neurones)

	(new_W, J_temp, E_temp) = back_propa(prediction=a, X_sample=X, y_sample=y, actual_W=w)
	w = new_W

	if(epoch%DIV==0):
		w_evo.append(w)

	J.append(J_temp.squeeze())

print("W_final : ",w) #[[-3.21  7.35  7.35]]
# W : (neurones, n+1)
# X_sample : (m, n+1)
print(w_evo)
plot_cost(J)
plot_dataset_and_model(dt=dataset, w_final=w, w_evolution=w_evo)






















def ReLU(iuts_datas):
	return max(iuts_datas, 0)

def forward_funtion(weights, biases, iuts_datas, activation_function):
	weights = array(weights)
	biases = array(biases)
	iuts_datas = array(iuts_datas)
	out = []

	if (iuts_datas.shape[0] == weights.shape[1] or weights.shape[0]==iuts_datas.shape[0] ):
		print("dimensions OK !\n")
	else:
		print("dimensions non corrects !\n")
		return 0

	out.append(dot(weights,iuts_datas))
	out = array(out).reshape( (weights.shape[0], 1) )

	biases = biases.reshape( (weights.shape[0], 1) )
	out += biases

	if(activation_function == "relu"):
		for i in range( len(out) ):
			out[i] = ReLU(out[i])

	elif(activation_function == "softmax"):
		pass		

	return out


def batch_gradient_descent(w_actual, learning_rate, z_actual, delta_previous):
	w_new = w_actual - learning_rate*z_actual*delta_previous
	return w_new

def delta_computation(w_previous, delta_previous, af_deriv_z_actual):
	delta_actual = w_previous*delta_previous*af_deriv_z_actual
	return delta_actual





