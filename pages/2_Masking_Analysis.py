import streamlit as st 
import cv2
import numpy as np
from PIL import Image

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 0.8rem;
                    padding-bottom: 0rem;
                    # padding-left: 2rem;
                    # padding-right:2rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)


def mask_image(image_array,masking_type,masked_img_col,masking_strength=20):

    if masking_type == "Pixelation":
        # Pixelate the bounding box area
        pixel_size = 5  # Adjust pixel size as needed
        pixelated_area = cv2.resize(image_array, (0, 0), fx=1/pixel_size, fy=1/pixel_size, interpolation=cv2.INTER_NEAREST)
        pixelated_area = cv2.resize(pixelated_area, (221 - 79, 135 - 46), interpolation=cv2.INTER_NEAREST)

        # Randomly color the pixels
        pixelated_image = np.random.randint(0, 256, size=pixelated_area.shape, dtype=np.uint8)  # Generate random colors
        pixelation_text = "Pixelation is a technique used to obscure details in an image by reducing its resolution. This is achieved by dividing the image into smaller blocks and replacing each block with a single color, resulting in a mosaic-like effect. This method is effective in masking sensitive information by making it less discernible. In this implementation, random colors are generated for each pixel to further obscure the details."
        return pixelated_image, pixelation_text
    
    elif masking_type == "Gaussian Blur":
        gaussian_blurred_image = cv2.GaussianBlur(image_array, (0, 0), masking_strength)
        guassian_text = "Gaussian Blur is a widely used technique in image processing for reducing image noise and detail. It works by applying a Gaussian function to the image, which smooths the image by averaging the pixel values with their neighbors. This results in a blurred effect that can help in masking sensitive information by making it less recognizable."
        return gaussian_blurred_image,guassian_text
    
    elif masking_type == "Mossaic Blur":
        height, width = image_array.shape[:2]
        temp_image = cv2.resize(image_array, (width // masking_strength, height // masking_strength), interpolation=cv2.INTER_LINEAR)
        mosaic_blurred_image = cv2.resize(temp_image, (width, height), interpolation=cv2.INTER_NEAREST)
        mosaic_text = "Mosaic Blur is a technique that reduces the image resolution by dividing it into smaller blocks and replacing each block with the average color of its pixels. This creates a mosaic effect that helps in masking sensitive information by making it less recognizable."
        return mosaic_blurred_image, mosaic_text
    
    elif masking_type == "Dithering":
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        dithered_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        dithered_image = cv2.cvtColor(dithered_image, cv2.COLOR_GRAY2BGR)
        dither_text = "Dithering is a technique used to create the illusion of color depth in images with a limited color palette. It works by scattering pixels of different colors to simulate intermediate shades. This method can help in masking sensitive information by adding noise to the image."
        return dithered_image, dither_text
    
    elif masking_type == "Watermarking":
        watermark_text = "MASK"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_thickness = 3
        text_size = cv2.getTextSize(watermark_text, font, font_scale, font_thickness)[0]
        text_x = (image_array.shape[1] - text_size[0]) // 2
        text_y = (image_array.shape[0] + text_size[1]) // 2
        watermarked_image = image_array.copy()
        cv2.putText(watermarked_image, watermark_text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)
        watermark_text = "Watermarking is a technique used to embed text or logos into an image to indicate ownership or confidentiality. This method helps in protecting sensitive information by clearly marking the image as confidential."
        return watermarked_image, watermark_text
    
    elif masking_type == "Black Masking":
        black_masked_image = image_array.copy()
        black_masked_image[:, :] = 0
        black_mask_text = "Black Masking is a technique used to completely obscure an image by covering it with a solid black color. This method is effective in masking sensitive information by making the entire image unrecognizable."
        return black_masked_image, black_mask_text
    return image_array


def masking_analysis_ui():

    # Loading the image from the path
    image = Image.open("pages/Masking_Test_Img.jpg")
    image_array = np.array(image)

    st.markdown(
            "<h1 style='text-align: left; font-size: 48px;'>Masking techniques Analysis</h1>",
            unsafe_allow_html=True,
        )

    st.markdown(
            "<p style='font-size: 20px; text-align: left;'>This module is specifically created to allow users to interactively explore various masking techniques commonly used for securing personal and confidential documents. On the right-hand side, detailed configuration instructions are provided to guide users through the interaction process. It's crucial to carefully review this information to ensure proper use and understanding of the module's functionalities. This interactive approach not only facilitates learning and experimentation with different masking methods but also emphasizes the importance of securely handling sensitive information.</p>",
            unsafe_allow_html=True,
        )
    
    masked_img_col = None
    col1,col2 = st.columns(spec=(2,1),gap = "large")
    with col1:
        st.write("")
        with st.container(border=True):
            original_img_col,masked_img_col = st.columns(spec=(1,1),gap = "large")
            with original_img_col:
                st.markdown("<h3 style='text-align: left; font-size: 28px;'>Original Image</h3>",
            unsafe_allow_html=True)
                st.image(image_array, use_column_width=True)
            

    with col2:
        st.markdown(
            "<p style='background-color: #C3E8FF; padding: 20px; border-radius: 10px; font-size: 20px;'>In this module, we will explore six distinct masking techniques, each offering unique methods for manipulating and processing images. To use this module, select a masking technique and press the Mask PD button. This will automatically invoke the corresponding function behind the scenes. Once executed, you will see the masked image displayed on the screen. This streamlined process ensures a seamless and efficient operation, making it user-friendly for various image processing tasks.</p>",
            unsafe_allow_html=True,
        )
        masking_option = st.selectbox(label="Select the Masking technqiue",options = ["Pixelation","Gaussian Blur","Mossaic Blur","Dithering","Watermarking","Black Masking"])
        mask_image_bt = st.button("Mask the image🔎",use_container_width=True)
        masking_strength = st.slider("Select the masking strength",min_value=1,max_value=50,value=20)

        if mask_image_bt:
            masked_img,text = mask_image(image_array[46:135,79:221],masking_option,masked_img_col,masking_strength)
            image_array[46:135,79:221] = masked_img
            with masked_img_col:
                st.markdown(f"<h3 style='text-align: left; font-size: 28px;'>Image after {masking_option}</h3>",
            unsafe_allow_html=True)
                st.image(image_array, use_column_width=True)
            with col1:
                st.markdown(
                    f"<p style='font-size: 20px; text-align: left;'>{text}</p>",
                    unsafe_allow_html=True,
                )



masking_analysis_ui()