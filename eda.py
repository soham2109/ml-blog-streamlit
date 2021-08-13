import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st












def app():
	st.title("Exploratory Data Analysis APP".upper())
	st.subheader("What is Exploratory Data Analysis (EDA)?")
	intro_to_EDA = """**Exploratory Data Analysis (EDA)** is a technique used by data scientists, data analysts and data engineers to analyze, investigate and summarize datasets, often employing data visualization methods.

There might be a lot of jargons in the definition, but things would get clear as we move along. For starters, let's just define a **dataset** to be a collection of values for different features. For instance, cars have different features like company, model name, engine fuel type, etc. and different cars have different values associated to these features.

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
	  - In the second graph,
	"""
	st.markdown(intro_to_EDA)

	col1, col2 = st.columns(2)
	with col1:
		st.markdown(anscombe_quartet_diag_html, unsafe_allow_html=True)
	with col2:
		st.markdown(anscombe_quartet_desc_html, unsafe_allow_html=True)

if __name__=="__main__":
	app()
