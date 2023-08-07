# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 01:18:29 2023

@author: USER
"""

# https://365datascience.com/tutorials/machine-learning-tutorials/how-to-deploy-machine-learning-models-with-python-and-streamlit/
# !pip install streamlit
# !pip install prediction
import streamlit as st 
import pandas as pd
import numpy as np
from PREDICTION import predict
import os
os.chdir('D:/2205 ALGOLINK/2307 한동대/LAB/')

st.title('Classifying Iris Flowers')
st.markdown('Toy model to play to classify iris flowers into \
setosa, versicolor, virginica')

st.header("Plant Features")
col1, col2 = st.columns(2)
with col1:
    st.text("Sepal characteristics")
    sepal_l = st.slider('Sepal lenght (cm)', 1.0, 8.0, 0.5)
    sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)
with col2:
    st.text("Petal characteristics")
    petal_l = st.slider('Petal lenght (cm)', 1.0, 7.0, 0.5)
    petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)
    
if st.button("Predict type of Iris"):
    result = predict(np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    if result[0] == 0:
        st.text('Based on the data your enter, this iris is Setosa')
    elif result[0] == 1:
        st.text('Based on the data your enter, this iris is Versicolor')
    else:
        st.text('Based on the data your enter, this iris is Virginica')

# st.write("""
#          # my first app hello *world!*
#          """)


# # Display text
# st.text('Fixed width text')
# st.markdown('_Markdown_') # see #*
# st.caption('Balloons. Hundreds of them...')
# st.latex(r''' e^{i\pi} + 1 = 0 ''')
# st.write('Most objects') # df, err, func, keras!
# st.write(['st', 'is <', 3]) # see *
# st.title('My title')
# st.header('My header')
# st.subheader('My sub')
# st.code('for i in range(8): foo()')

# # display data
# import pandas as pd
# df = pd.DataFrame({'a': [1,2,3],
#               'b': [2,3,4]})

# st.dataframe(df)
# st.table(df.iloc[0:10])

# # columns
# col1, col2 = st.columns(2)
# col1.write('Column 1')
# col2.write('Column 2')

# # Three columns with different widths
# col1, col2, col3 = st.columns([3,1,1])
# # col1 is wider

# # Using 'with' notation:
# with col1:
#      st.write('This is column 1')
# with col2:
#      st.write('This is column 2')
# with col3:
#      st.write('This is column 3')

# # tabs
# # Insert containers separated into tabs:
# tab1, tab2 = st.tabs(["Tab 1", "Tab2"])
# tab1.write("this is tab 1")
# tab2.write("this is tab 2")

# # You can also use "with" notation:
# with tab1:
#    st.radio('Select one:', [1, 2])

# st.json({'foo':'bar','fu':'ba'})
# st.metric(label="Temp", value="273 K", delta="1.2 K")

# # display media
# st.image('data/counselor.jpg')
# # st.audio(data)
# st.video(open('data/video.mp4','rb').read())