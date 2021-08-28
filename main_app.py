import eda
import lr
import svm
import knn
import mlp
import intro_to_ml
import streamlit as st

PAGES = { "Exploratory Data Analysis": eda,
		  "Intro to ML": intro_to_ml,
		  "Linear Regression": lr,
		  "Support Vector Machines": svm,
		  "k-Nearest Neighbours": knn,
		  "Multi-layer Perceptron": mlp}


def app():
	st.sidebar.title("Navigation")
	page_selected = st.sidebar.radio("Go to", list(PAGES))
	page = PAGES[page_selected]
	page.app()

if __name__ == "__main__":
	app()
