import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


np.random.seed(0)
random_state=11


@st.cache
def get_dataset():
	iris = datasets.load_iris()
	X = iris.data
	y = iris.target
	names = iris.target_names
	return X,y,names


def max_width(prcnt_width:int = 75):
    max_width_str = f"max-width: {prcnt_width}%;"
    st.markdown(f"""
                <style>
                .reportview-container .main .block-container{{{max_width_str}}}
                </style>
                """,
                unsafe_allow_html=True,
    )


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
	max_width(80)
	st.header("Multi-Layered Perceptron (MLP) Model")

	markdown = """
Before getting into what is multi-layered perceptron, we need to first understand what is **perceptron**. A perceptron model, in Machine Learning, is a supervised learning algorithm for binary classification. Based on a neuron in the human brain, perceptron acts as an artificial neuron that performs human-like brain functions. The perceptron uses a hyperplane to separate the input data into two classes. First conceptualized by Frank Rosenblat in 1957, it forms the basic building block for modern deep learning models. Perceptron enables machines to automatically learn the weights and biases. There are two perceptron models:

 - **Single-layered Perceptron**: It is defined by its ability to **linearly classify** inputs. This means that this model only utilizes a single hyperplane line and classifies the inputs as per the learned weights.
 - **Multi-layered Perceptron (MLP)**: It is a high processing algorithm that allows machines to classify inputs using more than one layers. The perceptron consists of an input layer and an output layer which are fully connected. MLPs have the same input and output layers but may have multiple hidden layers in between, and classifies datasets which are not linearly separable. It is also called a **feed-forward neural network**. The first layer is the _input layer_, the last layer is the _output layer_ and all other layers in between are called the _hidden layers_. The number of hidden layers and the number of neurons per layer, are called **hyperparameters** of the neural network, and they need tuning.

![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.ta1EwD8y5Jw781ckzQR9GQHaF3%26pid%3DApi&f=1)

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[Source](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.ta1EwD8y5Jw781ckzQR9GQHaF3%26pid%3DApi&f=1)

Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called **backpropagation** for training, using the gradient descent algorithm.


If a multilayer perceptron has a linear activation function in all neurons, i.e., a linear function that maps the weighted inputs to the output of each neuron, then linear algebra shows that any number of layers can be reduced to a two-layer input-output model. In MLPs some neurons use a nonlinear activation function that was developed to model the frequency of action potentials, or firing, of biological neurons. Since MLPs are fully connected, each node in one layer connects with a certain weight $w_{ij}$ to every node in the following layer.
	"""
	st.markdown(markdown)
	st.markdown("")
	st.markdown("")
	st.markdown("")

	markdown="""

## MLP Hands-on Tool

Below is an interactive tool, that allows you to understand how MLP parameters and hyper-paramters like number of hidden layers, number of neurons per hidden layer, the activation function, etc. affects its prediction, i.e. the separating hyperplane it creates.

There are knobs for the type of data one wants to fit the model to, i.e. whether the input data is clean, or if you want to introduce noise in the data, and see how it affects the prediction of the model.
	"""
	st.markdown(markdown)
	st.markdown("")
	st.markdown("")
	st.markdown("")



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
	prediction = list(clf.predict_test())
	st.write("The predicted class: {}".format(*prediction))

	st.markdown("")
	st.markdown("")
	references="""
## References:
  - Multi-layered Perceptron notes at [CMU](https://www.cs.cmu.edu/~10701/slides/multi-layer-perceptron_notes.pdf)
  - Wikipedia [article](https://en.wikipedia.org/wiki/Multilayer_perceptron) on Multi-Layered Perceptron
	"""
	st.markdown(references)



if __name__=="__main__":
	app()
