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
random_state=11

def max_width(prcnt_width:int = 75):
    max_width_str = f"max-width: {prcnt_width}%;"
    st.markdown(f"""
                <style>
                .reportview-container .main .block-container{{{max_width_str}}}
                </style>
                """,
                unsafe_allow_html=True,
    )



def get_data(dataset_type, num_points, noise=0, n_features=2):
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
	max_width(80)
	st.title("Support Vector Machines")

	st.header("What is Support Vector Machine?")
	markdown = """
Support Vector Machine is a **supervised** machine learning algorithm where one tries to find a hyperplane (a plane in higher-dimensional space) that best separates data into classes. **SVMs are among the best "off-the-shelf" supervised learning algorithm** (many consider it to be the best), that can be used for both classification and regression. SVM differs from logistic regression from the fact that although both try to find a separating hyperplane, SVM depends on statistical approaches rather than a probabilistic one.

For the purpose of understanding SVMs, we need to know the following:
  - **Support Vectors**: They are the data-points (dataset elements) nearest to the hyperplane, the points of the dataset that, if removed, would alter the dividing hyperplane. They are the **critical** elements of a dataset.
  - **Hyperplane**: For a simple classification task with two-classes and two features, a hypeplane is just a straight line that separates the two classes in the two-dimensional space. The more the points are away from the hyperplane, more is the confidence by which the points are classified. So, the objective of SVM is to fit such a hyperplane, such that data-points are at maximal separation from it, while still being on the correct side.
  - **Margin**: The distance between the hyperplane and the nearest data-pints (support vectors) is known as the margin. So, in simple terms, **SVM chooses the hyperplane with the greatest possible margin**. There are two types of margins:
    - Hard Margin
    - Soft Margin
  - **Kernels**: But sometimes, there is no clear separating boundary in the dataset, where kernels come into play. This projects the data into a higher dimensional space, where the data becomes linearly separable using something known aa the "Kernel Trick". There are different types of kernels:
    - Radial Basis Kernel
    - Polynomial Kernel
    - Sigmoid Kernel
  - **Hinge Loss**: The loss function that SVM minimizes, called the hinge-loss, if of the following form

     &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$\min \\frac{\gamma}{2}||W||^2 + C \sum_{i=1}^N \max(0, 1- y_i(x_i^TW + b))$

     where $\gamma$ is the regularization parameter, $C$ controls how hard the margin is for SVM. It represents the margin maximizing loss function.

	"""
	st.markdown(markdown)
	st.markdown("")
	st.markdown("")
	st.markdown("")

	markdown="""

## SVM Hands-on Tool

Below is an interactive tool, that allows you to understand how SVM parameters like _C_, $\gamma$, etc. affects its prediction, i.e. the separating hyperplane it creates.

There are knobs for the type of data one wants to fit the model to, i.e. whether the input data is linearly separable, or circular or moon shaped. Other options are if you want to introduce noise in the data, and see how it affects the prediction of the model.
	"""
	st.markdown(markdown)
	st.markdown("")
	st.markdown("")
	st.markdown("")


	st.subheader("Dataset")
	st.markdown("")
	st.markdown("")
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

	markdown="""

## References:
- Analytics Vidhya [blog](https://www.analyticsvidhya.com/blog/2021/10/support-vector-machinessvm-a-complete-guide-for-beginners/) on SVM

	"""
	st.markdown(markdown)


if __name__ == "__main__":
	app()
