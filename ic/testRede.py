from keras.models import Sequential
from keras.layers import Dense
import numpy as np
# fix random seed for reproducibility
np.random.seed(11)

limite = 100
number_sort = 200
X = []
Y = []
X_Eval = []
Y_Eval = []



def bubbleSort(alist, ct=False, ev=False, nn=False):
	for passnum in range(len(alist)-1,0,-1):
		for i in range(passnum):
			if alist[i]>alist[i+1]:
				if ct:
					if not ev:
						X.append([it/limite for it in alist.tolist()]+[passnum/9., i/9.])
					else:
						X_Eval.append([it/limite for it in alist.tolist()]+[passnum/9., i/9.])

				if not nn:
					temp = alist[i]
					alist[i] = alist[i+1]
					alist[i+1] = temp
				else:
					state = [it/limite for it in alist.tolist()]+[passnum/9., i/9.]
					result = model.predict(np.reshape(state, [1, len(state)]))[0]
					alist = result[:9]*limite
					# = int(result[9]*9)
					#i = int(result[10]*9)

				if ct:
					if not ev:
						Y.append([it/limite for it in alist.tolist()]+[passnum/9., i/9.])
					else:
						Y_Eval.append([it/limite for it in alist.tolist()]+[passnum/9., i/9.])
	return alist


def generate_data():
	for i in range(number_sort):
		print("Generating data ",i,"/",number_sort)
		aleatorio = np.random.sample(9)*limite
		bubbleSort(aleatorio, True)
		#print (X[len(X)-1])
		#print (Y[len(Y)-1])
	for i in range(int(number_sort*0.3)):
		print("Generating data ",i,"/",number_sort)
		aleatorio = np.random.sample(9)*limite
		bubbleSort(aleatorio, True, True)


# create model
model = Sequential()
model.add(Dense(200, input_dim=11, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(11, activation='sigmoid'))


# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Test NN
state = np.random.sample(11)
result = model.predict(np.reshape(state, [1, len(state)]))[0]
print (type(result))
print (result)
#exit()


#load model
try:
	model.load_weights("NN_swap.h5")
except OSError:
	# Fit the model
	print("Generating Data!")
	generate_data()
	print("Training")
	model.fit(X, X, epochs=20, batch_size=1)
	# evaluate the model
	scores = model.evaluate(X_Eval, X_Eval)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# save model
model.save_weights("NN_swap.h5")



print("\n\n\nResultado:")
alist = np.random.sample(9)*limite
print(alist)

state = [it/limite for it in alist.tolist()]+[0/9., 7/9.]
result = model.predict(np.reshape(state, [1, len(state)]))[0]
alist = result[:9]*limite

#alist = bubbleSort(alist, nn=True)

print(alist)

