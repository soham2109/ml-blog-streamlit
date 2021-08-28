import streamlit as st
from PIL import Image

def load_image(path):
	image = Image.open(path)
	image.resize((1000, 1000))
	return image

def app():
	st.title("Introduction to Machine Learning")
	st.markdown("---")
	st.header("Jargon Buster")
	markdown="""
There are so many buzzwords around, like *artificial intelligence* (AI), *machine learning* (ML), *deep learning* (DL), *data science* (DS), etc. and many use these terms interchangeably. But each of the terms have their own meaning, although their boundaries are not clearly defined. Let’s look at some of these terms briefly.
"""
	st.markdown(markdown)
	st.image(load_image("images/i01_ai_ml_dl_ds.png"),
				 caption="AI vs ML vs DL vs DS")
	col1, col2 = st.columns(2)
	with col1:
		markdown="""
**Artificial Intelligence** (AI):

Artificial Intelligence, fondly abbreviated as AI, is concerned with imparting human intelligence to machines. It focuses on the development of intelligent machines that can think and act like humans; essentially, AI is intelligence such as machines display. It deals with the following:
- General Intelligence
- Knowledge Representation
- Motion and Manipulation
- Reasoning and Problem Solving, etc.
"""
		st.markdown(markdown)
	with col2:
		markdown="""
**Machine Learning** (ML):

Machine Learning is concerned with giving machines the ability to learn by training algorithms on a huge amount of data. It makes use of algorithms and statistical models to perform a task without needing explicit instructions. Example usages:
- Detecting Spam
- Image Recognition
- Financial Analysis
- Recommendation Engines, etc.
"""
		st.markdown(markdown)

	col3, col4 = st.columns(2)
	with col3:
		markdown="""
**Deep Learning** (DL):

Deep Learning is an approach to Machine Learning; one that focuses on learning data representations rather than on task-specific algorithms. It makes use of Deep Neural Networks, which are inspired by the structure and function of the human brain. Example usages:
- Object Detection
- Image Inpainting/Reconstruction
- Predictive Systems, etc.
"""
		st.markdown(markdown)
	with col4:
		markdown="""
**Data Science** (DS):

Data Science is the term for a whole set of tools and techniques by which to analyze data and extract insights from it. It makes use of scientific methods, processes, and algorithms to make this happen. Data Science plays an important role in the following:
- Data Architecture
- Data Analysis
- Data Aquisition, etc.
	"""
		st.markdown(markdown)

	# Disclaimer
	st.markdown("This blog serves as an introduction to Machine Learning to the viewers. The other topics would hopefully be covered in future blogs.")


	st.header("Formal Definition of Machine Learning")
	st.markdown("""
According to **Tom Mitchell**, professor of Computer Science and Machine Learning at Carnegie Mellon, _a computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E._

Example: Detection of credit card fraud.
  - Task _T_ : assigning labels - fraud or not-fraud - to credit card transactions
  - Performance measure _P_ : Accuracy of the classification
  - Training experience _E_ : Historical data of credit card transactions labelled fraud or not.
	""")

	st.markdown("Next we move onto the 6 jars of machine learning, to describe the complete machine learning pipeline.")

	st.markdown("")
	st.markdown("")
	st.markdown("")
	st.markdown("")
	st.markdown("")

	# References for viewers
	st.header("References")
	markdown="""
- [Artificial Intelligence vs Machine Learning vs Deep Learning vs Data Science](https://data-flair.training/blogs/artificial-intelligence-vs-machine-learning-vs-dl-vs-ds/)
		"""
	st.markdown(markdown)


if __name__=="__main__":
	app()
