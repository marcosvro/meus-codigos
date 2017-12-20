from keras.models import Sequential
from keras.layers import Dense
import numpy as np
# fix random seed for reproducibility
#np.random.seed(0)

limite = 100
number_sort = 20000
X = []
Y = []
X_Eval = []
Y_Eval = []
X_Test_sort = []
Y_Test_sort = []


def bubbleSort(alist, ct=False, ev=False, nn=False):
	for passnum in range(len(alist)-1,0,-1):
		for i in range(passnum):
			if alist[i]>alist[i+1]:
				if ct:
					if not ev:
						X.append([it/limite for it in alist.tolist()]+[i/5.])
					else:
						X_Eval.append([it/limite for it in alist.tolist()]+[i/5.])

				if not nn:
					alist = vetor(alist, i, i+1)
				else:
					state = [it/limite for it in alist.tolist()]+[i/5.]
					result = model.predict(np.reshape(state, [1, len(state)]))[0]
					alist = np.rint(result[:5]*limite)
					# = int(result[5]*5)
					#i = int(result[10]*5)

				if ct:
					if not ev:
						Y.append([it/limite for it in alist.tolist()]+[i/5.])
					else:
						Y_Eval.append([it/limite for it in alist.tolist()]+[i/5.])
	return alist


def swap(vetor, i, j):
	temp = vetor[i]
	vetor[i] = vetor[j]
	vetor[j] = temp
	return vetor.tolist()

def generate_data():
	for i in range(number_sort):
		#print("Generating data ",i,"/",number_sort)
		aleatorio = np.random.sample(5)*limite
		aleatorio = np.rint(aleatorio)
		i = int(np.rint(np.random.random()*4))
		j = int(np.rint(np.random.random()*4))
		X.append((aleatorio/limite).tolist()+[i/4, j/4])
		Y.append([it/limite for it in swap(aleatorio, i, j)]+[i/4, j/4])
		#print (X[len(X)-1])
		#print (Y[len(Y)-1])
	for i in range(int(number_sort*0.3)):
		#print("Generating data ",i,"/",number_sort)
		aleatorio = np.random.sample(5)*limite
		aleatorio = np.rint(aleatorio)
		i = int(np.rint(np.random.random()*4))
		j = int(np.rint(np.random.random()*4))
		X_Eval.append((aleatorio/limite).tolist()+[i/4, j/4])
		Y_Eval.append([it/limite for it in swap(aleatorio, i, j)]+[i/4, j/4])


# create model
model = Sequential()
model.add(Dense(50, input_dim=7, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(7, activation='linear'))


# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

'''
# Test NN
state = np.random.sample(7)
result = model.predict(np.reshape(state, [1, len(state)]))[0]
print (type(result))
print (result)
#exit()
'''

print("Generating Data!")
generate_data()

#load model
try:
	model.load_weights("NN_swap.h5")
except OSError:
	# Fit the model
	print("Training")
	model.fit(X, Y, epochs=70, batch_size=10)
	# evaluate the model
	scores = model.evaluate(X_Eval, Y_Eval)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# save model
model.save_weights("NN_swap.h5")



print("\n\n\nResultado:")
for i in range(6):
	alist = np.rint(np.random.sample(5)*limite)
	print(alist)
	state = [it/limite for it in alist.tolist()]+[1/4, 4/4]
	result = model.predict(np.reshape(state, [1, len(state)]))[0]
	alist = np.rint(result[:5]*limite)
	print(alist,"\n")

num_acertos = 0
num_samples = len(X_Test_sort)
limiar_erro = 10
for i, x in enumerate(X_Test_sort):
	sorted_list_nn = bubbleSort(x,nn=True)
	dist = np.sum(np.sqrt((sorted_list_nn-Y_Test_sort)**2))
	if dist < limiar_erro:
		num_acertos += 1

print ("NUMEROS DE VETORES TESTADOS : ", num_samples)
print ("NUMEROS DE VETORES ORDENADOS: ", num_acertos)




