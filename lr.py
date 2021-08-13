import streamlit as st
import numpy as np
import mplcyberpunk
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression

plt.style.use("cyberpunk")
np.random.seed(0)

def plot_data(X,y, line_X, line_y):
	fig, ax = plt.subplots(1,1, figsize=(8,8))

	ax.scatter(X, y, s=50, marker='*', label = "Data Points", color = "yellowgreen")
	ax.plot(line_X, line_y, label="Regression Line", lw=2, color="red", alpha=0.8)
	ax.set_xlabel("Input")
	ax.set_ylabel("Response")
	ax.legend(loc="best")
	ax.patch.set_alpha(0)
	mplcyberpunk.add_glow_effects()
	return fig



def app():
	st.title("Simple Linear Regression")
	# Ask the user for the type of data to fit the regression model on

	col4, col5 = st.columns(2)
	with col4:
		dataset_type = st.selectbox("Select the type of data: ",["linear","polynomial"])
		if dataset_type=="polynomial":
			poly_degree = st.number_input("Enter the degree of polynomial: ",
										  format="%d",
										  min_value=1,
										  max_value=5,
										  value=1)
	# Ask the number of datapoints to regress on
	with col5:
		num_data_points = st.slider("Choose number of data points: ",
									min_value=10, max_value=300)


	# Ask if to include noise in the dataset
	col6, col7 = st.columns(2)
	with col6:
		use_noise = st.radio("Introduce noise in the dataset?",["No", "Yes"])
	if use_noise=="Yes":
		with col7:
			use_noise = True
			noise_std = st.slider("Select the standard deviation of noise: ",
								 min_value=0.0,
								 max_value=25.0)
	else:
		use_noise = False
		noise_std = 0

	# If the user has selected data-points and the number of data
	if dataset_type and num_data_points:
		bias = np.random.randint(-100, 100)
		if dataset_type=="linear":
			X,y,coef = datasets.make_regression(n_samples = num_data_points,
												n_features = 1,
												n_targets = 1,
												noise = noise_std,
												coef=True,
												bias = bias,
												random_state=0)
			coef = [coef]
		else:
			actual_coeffs = np.random.randint(-10, 10, poly_degree)
			X,y,coef = datasets.make_regression(n_samples = num_data_points,
												n_features = 1,
												n_targets = 1,
												noise = noise_std,
												coef=True,
												bias = bias,
												random_state=0)
			# y=np.zeros(y.shape)
			for i in range(1,poly_degree):
				y += actual_coeffs[i]*(np.power(X,i).flatten())
			actual_coeffs[0] = coef
			coef = list(actual_coeffs)

	# Fit the linear regression model on the dataset synthesized
	lr = LinearRegression()
	lr.fit(X,y)

	# Predict data of estimated models
	line_X = np.arange(X.min(), X.max()+1)[:, np.newaxis]
	line_y = lr.predict(line_X)



	# display the original coefficients and the predicted coefficients
	col8, col9 = st.columns(2)
	with col8:
		# get the plot of linear regression fit
		st.subheader("Get plot for regression")
		st.pyplot(plot_data(X, y, line_X, line_y))
	with col9:
		# st.text("\n\n\n")
		st.subheader("Performance of the Linear Regression model.")
		st.text("\n\n\n")
		st.text("The original coefficient(s) is(are):\nCoefficients: {},\nIntercept: {}".format(
												", ".join([str(np.round(i,2)) for i in coef]), bias))
		st.text("\n\n\n")
		st.text("The predicted coefficient(s) is(are):\nCoefficients: {}\nIntercept: {}".format(
													*np.round(lr.coef_,2),
													np.round(lr.intercept_,2)))
		st.text("\n\n\n")
		# display the overall score achieved using LinearRegression
		st.text("The R2-score achieved is: {}".format(lr.score(X, y)))



if __name__ == "__main__":
	app()
