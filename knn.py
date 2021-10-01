from collections import Counter
import random

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import matplotlib.colors as mplc
import mplcyberpunk

color_list = [k for k,v in mplc.cnames.items()]
# plt.style.use("bmh")
plt.style.use("cyberpunk")
NUM_CLUSTERS=2

np.random.seed(1001)
random_state=11

def normalize(X):
	# min-max normalizer
	return (X - X.min())/(X.max()-X.min())


def calc_distance(x1, x2):
	# euclidean distance measure desc
	return np.sqrt(np.sum(np.square(x1-x2), axis=1))


def find_k_nearest_neightbours(X, y, input_data, k):
	# given the dataset points and the unknown label points
	# and the neighbourhood to consider find the
	# k-nearest neighbours of the unknown point
	distance = calc_distance(X, input_data)
	indices = np.argsort(distance)[:k]
	return X[indices], y[indices]


def plot_data_points(X,y, input_data):
	# plot the data points and the unknown data
	fig, ax = plt.subplots(1,1,
						   constrained_layout=True,
						   figsize=(5,5))

	class1_data, class1_labels = X[y==1], y[y==1]
	class2_data, class2_labels = X[y==0], y[y==0]

	ax.scatter(class1_data[:,0],
				  class1_data[:,1],
				  color="darkred",
				  label="class1",
				  s=15,
				  alpha=0.7)
	ax.scatter(class2_data[:,0],
				  class2_data[:,1],
				  color="lightblue",
				  label="class2",
				  s=15,
				  alpha=0.7)
	ax.scatter(input_data[0],
				  input_data[1],
				  color="cyan",
				  label="unknown",
				  alpha=0.7,
				  s=40)

	ax.legend(loc="best")
	ax.set_ylim((-0.1, 1.1))
	ax.set_xlim((-0.1, 1.1))
	ax.set_title("Distribution of samples")
	return  fig


def plot_data_points_with_labels(X, y, X_nearest, y_nearest, input_data, label):
	fig, ax = plt.subplots(1,1,
						   constrained_layout=True,
						   figsize=(5,5))
	# select colors for k neigbours
	colors = random.sample(color_list, len(X_nearest))

	class1_data, class1_labels = X[y==1], y[y==1]
	class2_data, class2_labels = X[y==0], y[y==0]

	ax.scatter(class1_data[:,0],
				  class1_data[:,1],
				  color="darkred",
				  label="class1",
				  s=15,
				  alpha=0.7)
	ax.scatter(class2_data[:,0],
				  class2_data[:,1],
				  color="lightblue",
				  label="class1",
				  s=15,
				  alpha=0.7)
	ax.scatter(input_data[0],
				  input_data[1],
				  color="cyan",
				  # label="unknown",
				  s = 40,
				  alpha=0.7)
	plt.text(0.7, 0.05, 
		"Predicted Label: {}".format(label),
		fontsize=15)

	i = 0
	for x,y in zip(X_nearest, y_nearest):
		ax.plot([x[0], input_data[0]],
				[x[1], input_data[1]],
				colors[i],
				label="neighbour {}".format(i+1))
		i+=1

	ax.legend(loc="best")
	ax.set_ylim((-0.1, 1.1))
	ax.set_xlim((-0.1, 1.1))
	ax.set_title("Nearest Neighbour Prediction")
	return  fig



def get_labels(X, y, input_data, k):
	X_nearest, y_nearest = find_k_nearest_neightbours(X, y, input_data, k)
	label = Counter(y_nearest).most_common()[0][0]
	return label, X_nearest, y_nearest

@st.cache
def get_data(num_points=100):
	# generate 2-d synthetic data with 2 cluster centers
	X, y = make_blobs(n_samples = num_points,
					  n_features = 2,
					  centers = NUM_CLUSTERS,
					  cluster_std=4.1,
					  random_state=random_state)
	X = normalize(X)
	return X,y


def introduction():
	st.subheader("Introduction")
	markdown="""
The Nearest Neighbours (NN) algorithms are among the "simplest" of the supervised algorithms for the **task** of **Classification** (mostly, Regression is also possible using this algorithm). Nearest Neighbour is a special case of k-Nearest Neighbours (k-NN) algorithm where `k=1`. 

#### Characteristics of kNN algorithm

 - **Lazy** : In **k-NN** all the labelled-training instances are simply stored and no explicit learning takes place. Hence, k-NN algorithm is called **lazy** learning algorithm. 
During testing, the test-set instances are compared with the nearby training instances and the labels are consequently decided. 
(The idea of what **near** means is not quite clear at this point, but it will be clarified soon.)

 - **Local** : The basic principle of kNN model is that instead of learning the approximate mapping function $f(x)=y$ globally from the training data, kNN approximates a function **locally**, which depends on the testing data point as well as the training instances.
 - **Instance Based**: Since the prediction of the class of the test data depends on the comparison of a query of the data with the training set, kNN is called a **instance-based** learning.
 - **Non-parametric**: The kNN algorithm does not in any form make any assumptions about the training data, or the mapping function, hence, there are no parameters involved to learn.
 - Commonly used as a **discriminative** model.


 #### Prediction Algorithm for 1-Nearest Neighbour
 
 closest_point := None  
 closest_distance := $\inf$  
 for $i$= 1, $\ldots$, n:  
    &emsp;- current_distance := $d(x^{[i]}, x^{[q]})$  
 	&emsp;- if current_distance < closest_distance:   
 	&emsp;&emsp;	* closest_distance := current_distance  
 	&emsp;&emsp;	* target_value_of_closest_point := $y^{[i]}$   


 The prediction for 1-Nearest Neighbour algorithm is the target value of the closest point.
 The closest point by default is measured using **Euclidean Distance** (also called $L^2$ distance), that computes the distance between two points, $x^{[a]}$ and $x^{[b]}$ given by the formula:  
 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$d(x^{[a]}, x^{[b]})$ = $\sqrt{\sum_{j=1}^n (x^{[a]}_j- x^{[b]}_j)^2}$  


 In k-Nearest Neighbour there are two prediction algorithms:   
   - Plurality voting among the k nearest neighbors for classification, i.e. the **mode** of the target labels of the k-Nearest Neighbours is chosen to be the label of the target value.  
   - Averaging the continuous target variables of the $k$ nearest neighbors for regression. i.e. take the **mean** of the target values of the k-Nearest Neighbours.

#### Advantages:  
  - Simple algorithm and easy to compute
  - Powerful predictive algorithm as it does not approximate the global mapping from training features to the target space.

#### Disadvantages:
  - Suffers from the **curse of dimensionality**, which means that as the number of features of the data increases, so does the complexity in prediction.
  - If you are familiar with time-complexity of algorithms, k-NN in its naive version has the time-complexity of $O(nm)$ where $n$ stands for the number of training instances, whereas $m$ stands for the number of features of each data-point in the training set.
    But there are other implements that can be used to reduce the complexity (not discussed here).


#### Pseudocode for Naive k-NN (one of the possible approaches)
$\mathrm{D}_k := \{\}$  
while $|\mathrm{D}_k| < k$:  
	&emsp;- closest_distance := $\inf$  
	&emsp;- if current_distance < closest_distance:  
	&emsp;&emsp;	- closest_distance := current_distance  
	&emsp;&emsp;	- closest_point := $x^{[i]}$  
	&emsp;- add closest_point to $\mathrm{D}_k$  


	"""
	st.markdown(markdown)


def max_width_(prcnt_width:int = 75):
    max_width_str = f"max-width: {prcnt_width}%;"
    st.markdown(f""" 
                <style> 
                .reportview-container .main .block-container{{{max_width_str}}}
                </style>    
                """, 
                unsafe_allow_html=True,
    )


def app():
	max_width_(80)
	st.title("k-Nearest Neighbours Algorithm")

	introduction()

	st.subheader("Check the Nearest Neighbours algorithm for youself")
	st.markdown("")

	markdown="""
In the following dynamic example, one can experimentally observe the effect of choosing the value of $k$ in the k-Nearest Neighbour. Here, the dataset is generated is two-dimensional as it is for experimental purpose.
So, one can change the size of the training data, the feature values termed as _feature 1_ and _feature 2_ and the number of neighbours $k$ to be considered in prediction.
	"""
	st.markdown(markdown)
	st.markdown("")
	st.markdown("")
	col1, col2, col3, col4 = st.columns(4)
	with col1:
		num_points = st.slider("Choose size of training dataset",
						min_value = 10,
						max_value = 500,
						key="num_points")
	with col2:
		x_ = st.slider("Choose input feature 1",
						min_value = 0.0,
						max_value = 1.0,
						key="x")
	with col3:
		y_ = st.slider("Choose input feature 2",
						min_value = 0.0,
						max_value = 1.0,
						key="y")
	with col4:
		k = st.slider("Choose the number of neighbours.",
					  min_value=1,
					  max_value=9,
					  key="k")

	input_data = np.array([x_, y_])

	X, y = get_data(num_points)
	st.subheader("Visualize the Dataset and neighbours.")
	col1, col2 = st.columns(2)

	with col1:
		st.pyplot(plot_data_points(X, y, input_data))
	with col2:
		label, X_nearest, y_nearest = get_labels(X, y, input_data, k)
		# print(X_nearest.shape)
		st.pyplot(plot_data_points_with_labels(X, y, X_nearest, y_nearest, input_data, label))


if __name__=="__main__":
	app()
