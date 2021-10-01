import streamlit as st
import numpy as np
import mplcyberpunk
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression

plt.style.use("cyberpunk")
np.random.seed(0)
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


def introduction():
	st.header("What is Linear Regression?")
	markdown = """
Linear Regression is a **supervised** machine learning algorithm that is used to find the **linear** relationship (mapping) between the independent variables 
(input training features) and the dependent variable (the target value). It is of two types:  
  - Linear Regression : when the target value id dependent on one independent variable, i.e. of the form ($y = mx + c$).
    For **Linear Regression**, the dependent variable($y$) is related to the independent variable ($x$) by the form: $y$ = $\\beta_0 + \\beta_1x$  
where $\\beta_0$ is the intercept (bias) and $\\beta_1$ is the slope (i.e. the change in the dependent variable when the independent variable changes).
  - Multiple Regression : when there are multiple independent variables of which the dependent variable is a function of.
    For **Multiple Regression**, there are multiple independent variables on which the target variable is dependent on. So, if the input varables are $x_1, x_2, \ldots, x_n$
then the relationship between them and the dependent variable ($y$) is: $y$ = $\\beta_0 + \\beta_1x_1 + \\beta_2x_2 + \ldots + \\beta_nx_n$, where $\\beta_0$ is the intercept (or bias) 
and $\{\\beta_i\}_{i=1}^n$ are the coefficients.


The task is to find the best fit values for the intercepts and the coefficients, such that the **prediction error** (the difference between the predicted value 
and the actual (or true) value, also known as the **residue**) is minimized. 

The objective function (**residual sum of squares (RSS)**) thus can be written as $\\underset{\\beta}{min} \sum_{i=1}^n(y_i-\widehat{y}_i)^2$, where $\widehat{y}_i$ is the model predicted target value, 
$y_i$ is the true target value and $\\beta := \{\\beta_0, \\beta_1, \ldots, \\beta_n\}$ is the set of all parameters of the regression model.


#### Assumptions of Linear Regression
The formulation of linear regression depends on the following four assumptions:  
  - **Linearity** : The dependent variable $y$ should be linearly related to the independent variables $x$. This can be checked using the scatterplot approach.
  - **Normaliy** : The dependent variable $y$ and the error terms (residues) must possess a normal distribution (gaussian distribution with mean 0 and variance 1). This can be observed using histograms.
  - **Homoskedestacity** : The error terms must possess constant variance. This can be observed using a residual plot.
  - **Independence** : The error terms (residuals) must be uncorrelated i.e. error at $\epsilon_t$ must not indicate the at error at $\epsilon_{t+1}$. Presence of correlation in error terms is known as 
    **autocorrelation** and it drastically affects the regression coefficients and standard error values since they are assumed to be uncorrelated. Correlation matrix can be used to check this.
  - **No Multicollinearity** : There must be no correlation among independent variables. Presence of correlation in independent variables lead to Multicollinearity. If variables are correlated, it becomes extremely difficult for the model to determine the true effect of IVs on DV.
	"""
	st.markdown(markdown)


def app():
	max_width(80)
	st.title("Linear Regression")
	# Ask the user for the type of data to fit the regression model on

	introduction()

	st.markdown("")	
	st.markdown("")	
	st.markdown("")	


	col4, col5 = st.columns(2)
	with col4:
		dataset_type = st.selectbox("Select the type of data: ",["linear","polynomial"])
		if dataset_type=="polynomial":
			poly_degree = st.number_input("Enter the degree of polynomial: ",
										  format="%d",
										  min_value=1,
										  max_value=10,
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
												random_state=random_state)
			coef = [coef]
		else:
			poly_degree = int(poly_degree+1)
			actual_coeffs = np.random.randint(-10, 10, poly_degree)
			X,y,coef = datasets.make_regression(n_samples = num_data_points,
												n_features = 1,
												n_targets = 1,
												noise = noise_std,
												coef=True,
												bias = bias,
												random_state=random_state)
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
