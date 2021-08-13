import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

@st.cache
def get_dataset():
	iris = datasets.load_iris()
	X = iris.data
	y = iris.target
	names = iris.target_names
	return X,y,names


class mlp:
	def __init__(self, dataset, hidden_layer_size = 100,
				 activation="relu", max_iter=200, test_size=0.2):
		"""	MLPClassifier for streamlit
			:param dataset:			  the dataset of iris on which to train and validate
			:param hidden_layer_size: tuple to specify the #units in each layer
			:param activation: 		  the activation function for every layer
									  {"relu","sigmoid","tanh","identity"}
			:param max_iter: 		  maximum number of iterations
		"""
		self.X = dataset[0]
		self.y = dataset[1]
		self.names = dataset[2]

		self.sc=StandardScaler()
		self.sc.fit(self.X)
		self.X = self.sc.transform(self.X)

		self.hidden_layer_size = hidden_layer_size
		self.activation = activation
		self.max_iter = max_iter
		self.test_size = test_size

		self.shuffle = True
		self.random_state = 0
		self.mlp = MLPClassifier(hidden_layer_sizes=self.hidden_layer_size,
								 activation=self.activation,
								 max_iter=self.max_iter,
								 random_state=self.random_state)


	def train_mlp(self):
		X_train, X_test, y_train, y_test = self.split_dataset()
		self.mlp.fit(X_train, y_train)

		score = self.mlp.score(X_test, y_test)
		return score


	def split_dataset(self):
		X_train, X_test, y_train, y_test = train_test_split(self.X,
															self.y,
															test_size=self.test_size,
															stratify=self.y,
															shuffle=self.shuffle,
															random_state=self.random_state)
		return X_train, X_test, y_train, y_test


	def get_data(self):
		cols = st.columns(4)
		with cols[0]:
			sepal_length = st.slider('Sepal length',
									 min_value=.01,
									 max_value=8.0)
		with cols[1]:
			sepal_width = st.slider('Sepal width',
									 min_value=0.01,
									 max_value=4.0	)
		with cols[2]:
			petal_length = st.slider('Petal length',
									 min_value=0.01,
									 max_value=8.0)
		with cols[3]:
			petal_width = st.slider('Petal width',
									 min_value= 0.01,
									  max_value=4.0)

		features=None

		data = {'sepal_length': sepal_length,
				'sepal_width': sepal_width,
				'petal_length': petal_length,
				'petal_width': petal_width}
		features = pd.DataFrame(data, index=[0])

		st.subheader("Input Features:")
		st.write(features)
		return features.to_numpy()


	def predict_test(self):
		features = self.get_data()
		features = self.sc.transform(features)
		y_pred = self.mlp.predict(features)
		named_y = self.names[y_pred]
		return named_y


def app():
	st.header("Multi-Layered Perceptron Model")
	dataset = get_dataset()
	num_layers = st.slider("Choose number of hidden layers", min_value=2, max_value=5)

	sizes = []
	cols = st.columns(num_layers)
	for num in range(num_layers):
		with cols[num]:
			val = st.slider("Choose number of units in layer {}:".format(num+1),
							min_value=2, max_value = 50)
			sizes.append(val)

	# get all the hyperparameters for the network
	hidden_layer_size = tuple(sizes)

	#if hidden_layer_size:
	col1, col2 = st.columns(2)
	with col1:
		activation = st.radio("Choose the activation function:",
							  ["relu", "tanh", "logistic", "identity"])
	with col2:
		max_iter = st.slider("Choose the maximum number of iterations:",
							 min_value = 10,
							 max_value = 200)
		test_size = st.slider("Choose the %tage of data to be kept for validation:",
							  min_value=5,
							  max_value=50)
		test_size = test_size*1.0/100.0

#	train = st.button("Train the model")
#	if train:
	# generate the model and get the accuracy score
	clf = mlp(dataset, hidden_layer_size, activation, max_iter, test_size)
	score = clf.train_mlp()
	st.write("The Accuracy score of the model: {}%".format(round(score*100,3)))

	# ask for a input from the user
	# test = st.button("Test your model")
	# if test and clf:
	prediction = clf.predict_test()
	st.write("The predicted class: {}".format(prediction))



if __name__=="__main__":
	app()
