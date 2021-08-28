import streamlit as st
from PIL import Image

def load_image(path):
	image = Image.open(path)
	return image

def app():
	st.title("Introduction to Machine Learning")
	st.markdown("---")
	st.subheader("Jargon Buster")
	markdown="""
There are so many buzzwords around, like *artificial intelligence* (AI), *machine learning* (ML), *deep learning* (DL), *data science* (DS), etc. and many use these terms interchangeably. But each of the terms have their own meaning and although their boundaries are not clearly defined. Let’s look at some of these terms with the help of the following diagram:
"""
	st.markdown(markdown)
	col1, col2 = st.columns(2)
	with col1:
		st.image(load_image("images/i01_ai_ml_dl_ds.png"),
				 caption="AI vs ML vs DL vs DS")
	with col2:
		markdown="""
- **Artificial Intelligence** (AI):
- **Machine Learning** (ML):
- **Deep Learning** (DL):
- **Data Science** (DS):
		"""
		st.markdown(markdown)


if __name__=="__main__":
	app()
