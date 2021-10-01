import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def max_width(prcnt_width:int = 75):
    max_width_str = f"max-width: {prcnt_width}%;"
    st.markdown(f""" 
                <style> 
                .reportview-container .main .block-container{{{max_width_str}}}
                </style>    
                """, 
                unsafe_allow_html=True,
    )

def app():
	max_width(80)
	st.title("Exploratory Data Analysis".upper())
	st.subheader("What is Exploratory Data Analysis (EDA)?")
	intro_to_EDA = """**Exploratory Data Analysis (EDA)** is a technique used by data scientists, data analysts and data engineers to analyze, investigate and summarize datasets, often employing data visualization methods.

Let's just define a **dataset** to be a collection of values for different features. For instance, cars have different features like company, model name, engine fuel type, etc. and different cars have different values associated to these features.

To explain the need for EDA, let's look into a famous dataset called the [**Anscombe Quartet**](https://en.wikipedia.org/wiki/Anscombe%27s_quartet). It consists of four datasets having the same data mean and variance. So, statistical methods like moment calculations, are incapable of distinguishing the actual difference between the datasets."""

	anscombe_quartet_diag_html = """
	<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Anscombe%27s_quartet_3.svg/425px-Anscombe%27s_quartet_3.svg.png" width="100%" height="100%"></img>
	<br>
	<center>Source: <a href="https://en.wikipedia.org/wiki/Anscombe%27s_quartet">Wikipedia</a></center>"""
	anscombe_quartet_desc_html = """
	- For all the four datasets:
	  - Mean of x: 9
	  - Mean of y: 7.50
	  - Variance of x: 11
	  - Variance of y: 4.125 &#177 0.33

	- From the graphs, we can conclude that:
	  - In the first graph (with x_1 and y_1), y_1 looks to be linearly dependent on x_1 with some noise.
	  - In the second graph (with x_2 and y_2), the values are non-linear (probably quadratic).
	  - In the third graph (with x_3 and y_3), the points are mostly linear with an outlier.
	  - In the final graph (with x_4 and y_4), the values are independent of x_4 with one outlier.
	"""
	st.markdown(intro_to_EDA)

	col1, col2 = st.columns(2)
	with col1:
		st.markdown(anscombe_quartet_diag_html, unsafe_allow_html=True)
	with col2:
		st.markdown(anscombe_quartet_desc_html, unsafe_allow_html=True)

	data="""One could not make out these simple deductions just by looking into the data points, but visualization makes it possible to convey these facts. In machine learning as well, we deal with high-dimensional data points but the interdependence between the features are not visible from the data-points.

This is where EDA comes into place. To completely understand the data provided to you, plotting the dependencies is necessary. We can infer the dependencies, distributions of the features in the dataset to further understand the problem at hand and select a suitable model. If there are some redundant features in the dataset, then they can be dropped to remove redundancy, and new features can be prepared from existing features to make the model more robust. This is what we commonly call **pre-processing** of data.

This falls under the **Data** jar of the 6 Jars of Machine Learning recipe.
	"""
	st.markdown(data)

	st.markdown("")
	references="""
- Wikipedia Article on [Anscombe Quartet](https://en.wikipedia.org/wiki/Anscombe%27s_quartet)
	"""
	st.header("References")
	st.markdown(references)


if __name__=="__main__":
	app()
