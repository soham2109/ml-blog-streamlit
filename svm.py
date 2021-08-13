import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import (make_moons,
							  make_circles,
							  make_classification)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import mplcyberpunk
from matplotlib.colors import ListedColormap

np.random.seed(0)
plt.style.use("cyberpunk")


def get_data(dataset_type, num_points, noise=0, random_state=0, n_features=2):
	if dataset_type=="Moons":
		X,y = make_moons(noise=noise,
						 random_state=random_state,
						 n_samples = num_points)
	elif dataset_type=="Circles":
		X,y = make_circles(noise = noise,
						   random_state=random_state,
						   n_samples = num_points)
	else:
		X,y = make_classification(n_features = n_features,
								  random_state = random_state,
								  n_samples = num_points,
								  n_redundant=0,
								  n_informative=n_features,
								  n_clusters_per_class = 1)
		if noise:
			X += 3*np.random.randn(*X.shape)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
	return X, y, X_train, X_test, y_train, y_test


def plot_data(X, y, X_train, X_test, y_train, y_test, \
			  classifier, classifier_name, score, h=0.02):
	x_min = X[:,0].min()-0.5
	x_max = X[:,0].max()+0.5

	y_min = X[:,1].min()-0.5
	y_max = X[:,1].max()+0.5

	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
						 np.arange(y_min, y_max, h))
	cm = ListedColormap(["r", "b"])
	fig, ax = plt.subplots(1,2, constrained_layout=True)

	# plot the training and testing data in the first column
	ax[0].scatter(X_train[:,0], X_train[:,1],
				  c=y_train, edgecolors='k', cmap=cm,
				  label="Training Data")
	ax[0].scatter(X_test[:,0], X_test[:,1],
				  c=y_test, edgecolors='k', cmap=cm,
				  alpha=0.5, label="Test Data")
	ax[0].set_xlim(xx.min(), xx.max())
	ax[0].set_ylim(yy.min(), yy.max())
	ax[0].legend(loc="best")
	# ax[0].set_xticks(())
	# ax[0].set_yticks(())

	# plot the classifier decision boundary
	if hasattr(classifier, "decision_function"):
		zz = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
	else:
		zz = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

	zz = zz.reshape(xx.shape)
	ax[1].contourf(xx, yy, zz, cmap=plt.cm.RdBu, alpha=0.5)
	ax[1].scatter(X_train[:,0], X_train[:,1],
				  c=y_train, edgecolors='k', cmap=cm,
				  label="Training Data")
	ax[1].scatter(X_test[:,0], X_test[:,1],
				  c=y_test, edgecolors='k', cmap=cm,
				  alpha=0.5, label="Test Data")
	ax[1].set_xlim(xx.min(), xx.max())
	ax[1].set_ylim(yy.min(), yy.max())
	# ax[1].set_xticks(())
	# ax[1].set_yticks(())
	ax[1].set_title("Kernel: {}".format(classifier_name))
	ax[1].legend(loc="best")
	ax[1].text(xx.max() - .3, yy.min() + .3, "Score: {:.2f}".format(score),
			   size=12, horizontalalignment='right', color="darkred")

	return fig


def app():
	st.title("Support Vector Machines")
	st.subheader("Dataset")
	col1, col2 = st.columns(2)
	with col1:
		dataset_type = st.selectbox("Choose the dataset type to train the SVM Classifier on:",
									["Linearly Separable", "Moons", "Circles"])
	with col2:
		num_points = st.slider("Choose the number of data points: ",
								min_value=10, max_value=300)

	col3, col4 = st.columns(2)
	with col3:
		use_noise = st.radio("Introduce noise in the dataset?",["No", "Yes"])
	if use_noise=="Yes":
		with col4:
			use_noise = True
			noise_std = st.slider("Select the standard deviation of noise: ",
								 min_value=0.0,
								 max_value=5.0)
	else:
		use_noise = False
		noise_std = 0

	X, y, X_train, X_test, y_train, y_test= get_data(dataset_type,
				   									 num_points,
				   									 noise=noise_std)

	kernel_dict={"Linear": "linear",
				 "Radial Basis Function": "rbf",
				 "Polynomial": "poly",
				 "Sigmoid": "sigmoid"}

	st.subheader("Choose Support Vector Machine Parameters")
	col5, col6, col7 = st.columns(3)
	with col5:
		kernel_type = st.radio("Choose the type of kernel: ",
							   ["Linear", "Radial Basis Function", "Polynomial", "Sigmoid"])
		kernel = kernel_dict[kernel_type]
	with col6:
		C = st.slider("Choose the value of C:", min_value=0.1, max_value=5.0)
	with col7:
		gamma = st.slider("Choose the value of gamma:", min_value=0.001, max_value=5.0)

	classifier = SVC(kernel=kernel,
					 random_state=0,
					 C=C,
					 gamma=gamma)
	classifier.fit(X_train, y_train)
	score = classifier.score(X_test, y_test)

	st.subheader("Visualizing the Decision Boundaries using {} kernel".format(kernel_type))
	st.pyplot(plot_data(X, y, X_train, X_test, y_train, y_test,
			  			classifier, kernel_type, score, h=0.02))



if __name__ == "__main__":
	app()
