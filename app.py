import streamlit as st
import utils
import pandas as pd
import centernet
import copy
from PIL import Image
import numpy as np
import warnings
import cv2
warnings.filterwarnings("ignore")

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def predict(image):
    model = centernet.build()
    image = cv2.resize(image,(1024,1024))
    hm, regr = model.pred_on_img(image)
    return image, hm, regr
    
@st.cache(suppress_st_warning=True, allow_output_mutation=True) 
def update_image_state(images_uploaded):
    st.session_state.images = images_uploaded
    st.session_state.curr_idx = 0

def generate_classification_report(images_uploaded):
    return images_uploaded

#Initialize Session State s   
if 'curr_idx' not in st.session_state:
    st.session_state.curr_idx = 0
    
if 'images' not in st.session_state:
    st.session_state.images = None
    
st.title("Red Blood Cell Detection and Analysis")

with st.sidebar:

    st.write('## File Upload')
    
    with st.form('upload', clear_on_submit=True):
        images_uploaded =  st.file_uploader('', type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
        submitted = st.form_submit_button("Upload and Compute!")
        update_image_state(images_uploaded)

        if submitted:
            if (st.session_state.images is None) or (len(st.session_state.images) == 0):
                st.write("Please upload files first")
            else:
                st.write("Success")
    
if (st.session_state.images is not None) and (len(st.session_state.images)>0):

    image_names = {image.name:i for i,image in enumerate(st.session_state.images)}

    if len(st.session_state.images)>1:
        
        col1, col2, col3 = st.columns(3)

        with col1:
            Prev = st.button('Previous Image')
        with col2:
            Report = st.button('Generate Report')
        with col3:
            Next = st.button('Next Image')
            
        if Prev:
            st.session_state.curr_idx = (st.session_state.curr_idx-1)%len(st.session_state.images)
            index = image_names[st.sidebar.selectbox('Please select your file', list(image_names.keys()), index=copy.deepcopy(st.session_state.curr_idx))]
        elif Next:
            st.session_state.curr_idx = (st.session_state.curr_idx+1)%len(st.session_state.images) 
            index = image_names[st.sidebar.selectbox('Please select your file', list(image_names.keys()), index=copy.deepcopy(st.session_state.curr_idx))]
        else:
            index = image_names[st.sidebar.selectbox('Please select your file', list(image_names.keys()), index=copy.deepcopy(st.session_state.curr_idx))]
            st.session_state.curr_idx = copy.deepcopy(index)

        image_uploaded = st.session_state.images[st.session_state.curr_idx]
        
    else:
        Report = st.button('Generate Report')
        image_uploaded = st.session_state.images[0]
    
    st.write(f'### Currently Displaying : {image_uploaded.name}')

    st.sidebar.write('## Object Detection Parameters')
    conf = st.sidebar.slider(f'Confidence Threshold', 0.0, 1.0, 0.5)   
    iou = st.sidebar.slider(f'Overlapping IOU', 0.0, 1.0, 0.5)
    width = st.sidebar.slider(f'Set Image Size', 200, 1000, 600)

    image = Image.open(image_uploaded).convert('RGB')
    image = np.array(image)
    
    image, hm, regr = predict(image)
    show = copy.deepcopy(image)
    show, count = utils.fullPLOT(show, hm, regr, conf, iou)
    
    st.write(f'#### Found {count} Red Blood Cells')
    placeholder = st.image(show, width=width)

    if Report:
        dataframe = pd.DataFrame()
        counts = []
        names = []

        for image_uploaded in st.session_state.images:
            image = Image.open(image_uploaded).convert('RGB')
            image = np.array(image)
            image, hm, regr = predict(image)
            show = copy.deepcopy(image)
            show, count = utils.fullPLOT(show, hm, regr, conf, iou)
            name = image_uploaded.name
            counts.append(count)
            names.append(name)

        dataframe['File Name'] = names
        dataframe['RBC Count'] = counts
            
        st.dataframe(dataframe)

        csv = dataframe.to_csv().encode('utf-8')

        st.download_button(
            "Download Full Report as CSV",
            csv,
            "rbc_analysis_report.csv",
            "text/csv",
            key='download-csv'
            )

    