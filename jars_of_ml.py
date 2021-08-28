import streamlit as st
from PIL import Image

def load_image(path):
	image = Image.open(path)
	image.resize((1000, 1000))
	return image


def app():
	# & jars of Machine Learning
	st.title("6 jars of Machine Learning")
	st.image(load_image("images/6_jars_of_ml.png"),
				 caption="Source: One Fourth Labs")
	markdown="""
There are six jars (or ingredients or elements) to any machine learning problem. They are:

1. **Data** : Machine Learning heavily depends on the quality of the structured dataset at hand. So, proper pre-processing, analysis and cleaning of data is required before proceeding with machine learning models, which is achieved through exploratory data analysis (**EDA**). Data can be curated in the following ways:

 - Open-source datasets present at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php), [Kaggle](www.kaggle.com).
 - Outsource through paid services like Amazon Mechanical Turk, etc.
 - Create your own dataset (Data Acquisition)

2. **Task** : Once the dataset to be used has been identified and cleaned, the next step is to define the tasks to be solved. The tasks can be as simple as weather forecasting, or as complicated as finding cancerous cells in medical images. So, accordingly tasks can be divided into two types:
  - **Supervised** : In these tasks, there is a requirement for both input (features) and output (labels) data to be supplied to the machine learning model. Here, the model is trained based on the labels. There are two types of supervised tasks:

    - **Regression** : It is a statistical method that attempts to determine the strength and character of the relationship between one dependent variable (output) and a set of independent variables (input).

    - **Classification** : It is the problem of predicting a value from a finite set (assuming that the output labels are from a finite set) from the given input data. It is a process to group data together based on the input features.

  - **Un-supervised** : In unsupervised learning problems, all input is unlabelled and the algorithm must create structure out of the inputs on its own. Clustering problems (or cluster analysis problems) are unsupervised learning tasks that seek to discover groupings within the input datasets.
Examples of this could be patterns in stock data or consumer trends.

3. **Model** :
4. **Loss** :
5. **Learning** :
6. **Evaluation** :

	"""
	st.markdown(markdown)

	st.markdown("")
	st.markdown("")
	st.markdown("")
	st.markdown("")
	st.markdown("")

	# References for viewers
	st.header("References")
	markdown="""
- [One Fourth Labs course on Deep Learning](https://padhai.onefourthlabs.in/courses/dl-feb-2019)
		"""
	st.markdown(markdown)


if __name__=="__main__":
	app()
