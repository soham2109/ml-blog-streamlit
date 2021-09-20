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

 Also the data in machine learning can be divided according to structure. Data can be **structured** (i.e. having a proper structure as in databases, records, etc.) or it can be **unstructured** (e.g. facebook, PDFs, tweets, etc.).

 Once the data has been curated, the complete data has to be further divided into two separates datasets (can have the output labelled or unlabelled). They are:
   - **Training Dataset** : The set of data-points in the data on which the model would be trained on.
   - **Validation Dataset** : The set of data-points on which the model will be validated, i.e. to check if the parameters and the hyperparameters chosen gives a satisfactory measure of accuracy.
   - **Test Dataset** : The set of unknown data-points on which the trained model that is used to predict after deployment.


2. **Task** : Once the dataset to be used has been identified and cleaned, the next step is to define the tasks to be solved. The tasks can be as simple as weather forecasting, or as complicated as finding cancerous cells in medical images. So, accordingly tasks can be divided into two types:
    - **Supervised** : In these tasks, there is a requirement for both input (features) and output (labels) data to be supplied to the machine learning model. Here, the model is trained based on the labels. There are two types of supervised tasks:

      - **Regression** : It is a statistical method that attempts to determine the strength and character of the relationship between one dependent variable (output) and a set of independent variables (input).

      - **Classification** : It is the problem of predicting a value from a finite set (assuming that the output labels are from a finite set) from the given input data. It is a process to group data together based on the input features.

    - **Un-supervised** : In unsupervised learning problems, all input is unlabelled and the algorithm must create structure out of the inputs on its own. Clustering problems (or cluster analysis problems) are unsupervised learning tasks that seek to discover groupings within the input datasets. Examples of this could be patterns in stock data or consumer trends.

3. **Model** : A machine learninng model is the output of the training process and is defined as the mathematical representation of the real-world process. The machine learning algorithms find patterns in the training dataset, which is used to approximate the target function and is responsible for mapping inputs to outputs from the available training dataset.

 Let ![\mathfrak{D}={(X_i,y_i)_{i=1}^N}](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B80%7D%20%5Cbg_black%20%5Csmall%20%5Cmathfrak%7BD%7D%3A%3D%5C%7B%28X_i%2Cy_i%29_%7Bi%3D1%7D%5EN%5C%7D%20%5Ctext%7B%20where%20%7D%20X_i%20%5Ctext%7B%20is%20the%20data%20and%20%7D%20y_i%20%5Ctext%7B%20is%20the%20target%20of%20the%20%7D%20i%5E%7Bth%7D%20%5Ctext%7B%20datapoint%20in%20the%20dataset%7D)

 The model finds a mathematical relationship (function) between the datapoints and the targets. As the number of parameters in the model increases the function becomes more complex and the model is then chosen on the basis of a concept called **Bias - Variance** tradeoff, which would have the optimum parameters and gives the lowest error as well.

4. **Loss** : Training a model simply means learning (determining) optimum values for all parameters from training examples. In machine learning, the goal is to find a model that minimizes **loss** by a process called **empirical risk minimization**. Loss is the penalty (calculated by a **loss function** depending on the application) for a bad prediction, i.e. if the model predicts the target value correctly (ideal case) the loss is zero, else it is greater. Following are some examples of loss functions:
   - Classification:
     - Log-Loss
     - Focal Loss
     - KL-Divergence / Cross-Entropy Loss
     - Hinge Loss, etc.
   - Regression:
     - Mean Squared Error (MSE) Loss
     - Mean Absolute Error (MAE) Loss
     - Huber Loss (smoothed MAE Loss), etc.

  Do not get overwhelmed by the names of the loss functions. These would get clarified as we look into the machine learning algorithms.

5. **Learning** : Now that have a vague idea of what the model, data, task and the loss functions are for our machine learning recipe, lets move onto what the learning in machine learning stand for. The machine learning model has to have the optimum parameters, which is actually a search problem and is solved by some optimization techniques, which vary according to the problem at hand. Some of the great solvers are:
   - Gradient Descent
     - Batch Gradient Descent
     - Stochastic Gradient Descent
     - Mini-Batch Gradient Descent
   - AdaGrad (based on Gradient Descent)
     - RMSProp
     - Adam,etc.

   Depending on the type of the problem at hand, one of the solvers is used in training the machine learning model depending on the loss function calculation and model parameters.

6. **Evaluation** : This is the final stage of the machine learning model recipe where we have already made the model learn the relationship between the inputs and targets using the learning algorithm and loss function, to have the optimal parameters. This stage is used for hyperparameter tuning, as we test the model on some unseen data (mostly **validation** dataset). This step helps in making the model more generalized. Some of the evaluation metrics that can be used for evaluation are:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Top-K Accuracy, etc.

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
- [Six Jars of Machine Learning](https://medium.datadriveninvestor.com/six-jars-of-machine-learning-2dd5a72ca1b)
		"""
	st.markdown(markdown)

#	# to hide the made with streamlit footer
#	hide_footer_style = """
#	<style>
#	.reportview-container .main footer {visibility: hidden;}
#	"""
#	st.markdown(hide_footer_style, unsafe_allow_html=True)


if __name__=="__main__":
	app()
