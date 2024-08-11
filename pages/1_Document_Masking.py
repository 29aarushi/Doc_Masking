import streamlit as st
import cv2
import zipfile
from PIL import Image
import io
import sys
import numpy as np
from ultralytics import YOLO
import time
import zipfile


st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    # padding-left: 2rem;
                    # padding-right:2rem;
                }
                .top-margin{
                    margin-top: 4rem;
                    margin-bottom:2rem;
                }
                .block-button{
                    padding: 10px; 
                    width: 100%;
                    background-color: #c4fcce;
                }
        </style>
        """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    """
    Load the pre-trained YOLO models for number plates and personal cards.

    This function loads two YOLO models:
    1. A model trained to detect number plates.
    2. A model trained to detect personal cards.

    Returns:
        tuple: A tuple containing the loaded number plate model and personal cards model.
    """
    # Loading the number plate and personal cards trained model
    np_model = YOLO(r"artifacts\Number_Plate\best.pt")
    pc_model = YOLO(r"artifacts\Personal_Cards\best.pt")
    return np_model, pc_model



def extract_zip_files(zip_file, file_type="image"):
    """
    Extract image or video files from a zip archive.

    This function extracts all files with .jpg or .png extensions if file_type is "image",
    or all files with .mp4 extensions if file_type is "video" from the given zip file.

    Args:
        zip_file (str): Path to the zip file containing images or videos.
        file_type (str): Type of files to extract ("image" or "video").

    Returns:
        list: A list of file names extracted from the zip file.
    """
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        if file_type == "image":
            files = [
                zip_ref.read(f)
                for f in zip_ref.namelist()
                if f.endswith(".jpg") or f.endswith(".png")
            ]
        elif file_type == "video":
            files = [
                f
                for f in zip_ref.namelist()
                if f.endswith(".mp4")
            ]
        else:
            raise ValueError("Invalid file_type. Use 'image' or 'video'.")
    return files





def save_images_to_zip(image_list):

    # Create an in-memory ZIP file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w") as zip_file:
        # Save each NumPy array as an image file and add it to the ZIP archive
        for i, image_array in enumerate(image_list):
            # Convert the NumPy array to a PIL Image object
            image = Image.fromarray(image_array)

            # Save the image to an in-memory BytesIO object
            image_buffer = io.BytesIO()
            image.save(image_buffer, format="PNG")
            image_buffer.seek(0)  # Reset the buffer pointer

            # Add the image to the ZIP archive
            filename = f"image_{i}.png"
            zip_file.writestr(filename, image_buffer.read())

    # Reset the buffer pointer
    zip_buffer.seek(0)
    return zip_buffer



def yolo_page():

    
    np_model, pc_model = load_models()
    images_list = None
    video_list = None
    
    output_col, config_col = st.columns([2.3, 1], gap="large")
    with output_col:
        st.markdown(
            "<h1 style='text-align: left; font-size: 55px;'>Document Masking</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size: 22px; text-align: left;'>Our aim is to provide a seamless solution for identifying personal documents and ensuring their confidentiality through effective masking techniques. In today's digital age, safeguarding personal information is paramount, and our application is designed to assist individuals and organizations in achieving this goal.</p>",
            unsafe_allow_html=True,
        )
        st.write("***")

        orginal_col,masked_col = st.columns([1,1], gap="large")
       

    with config_col:
        st.write("")
        st.write("")
        st.markdown(
            "<p style='background-color: #C3E8FF; padding: 20px; border-radius: 10px; font-size: 20px;'>Before starting the detection and redaction process, upload a compressed zip folder containing all the query images below. The zip folder will be unpacked, and the images will be extracted for analysis. Next, upload the query image for masking. The trained YOLOv8 model will detect personal documents, generate bounding boxes, and apply pixelation masking to those regions. The masked image will be displayed, along with a count of personal and non-personal documents detected.</p>",
            unsafe_allow_html=True, 
        )
        

        selected_image = st.selectbox(
            "Select Test Image",
            ("1 Query Card", "2 Query Cards", "4 Query Cards", "5 Query Cards")
        )
        display_bt = st.button("Show Orignal/Masked Image",use_container_width=True)
        if display_bt:
            if selected_image == "1 Query Card":
                with orginal_col:
                    st.image("pages/Test_Image_Results/Original/1_Card_Original.jpeg",width=400)
                with masked_col:
                    st.image("pages/Test_Image_Results/After Masking/1_Card_Masked.jpeg",width=400)
            elif selected_image == "2 Query Cards":
                with orginal_col:
                    st.image("pages/Test_Image_Results/Original/2_Cards_Original.jpeg",width=400)
                with masked_col:
                    st.image("pages/Test_Image_Results/After Masking/2_Cards_Masked.jpeg",width=400)
            elif selected_image == "3 Query Cards":
                with orginal_col:
                    st.image("pages/Test_Image_Results/Original/3_Cards_Original.jpeg",width=400)
                with masked_col:
                    st.image("pages/Test_Image_Results/After Masking/3_Cards_Masked.jpeg",width=400)
            elif selected_image == "4 Query Cards":
                with orginal_col:
                    st.image("pages/Test_Image_Results/Original/4_Card_Original.jpeg",width=400)
                with masked_col:
                    st.image("pages/Test_Image_Results/After Masking/4_Cards_Masked.jpeg",width=400)
            elif selected_image == "5 Query Cards":
                with orginal_col:
                    st.image("pages/Test_Image_Results/Original/5_Cards_Original.jpeg",width=400)
                with masked_col:
                    st.image("pages/Test_Image_Results/After Masking/5_Cards_Masked.jpeg",width=400)

           

yolo_page()
