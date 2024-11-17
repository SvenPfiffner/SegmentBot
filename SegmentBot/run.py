import streamlit as st
from PIL import Image
import torch
import psutil
import numpy as np
from PIL import Image
from editors import Editor
from processor import process_image, ProcessorSettings

@st.dialog("Disclaimer", width="large")
def disclaimer_modal():
    st.write("This app is free software and may contain bugs or other issues. By using this app, you agree that the developers are not responsible for any issues that may arise from using it. All functionality is provided as is and without guarantees.")
    st.write("Images uploaded to this app are not stored or evaluated beyond the processing required for the app to function.")
    st.write("This app is published with the intent to be used for personal and research use and should not be used for any illegal activities or to harm others. For detailed information on commercial use, please refer to the official source code repository linked on the main page.")
    st.write("By performing any action on this app, you confirm that you have read, understood and agreed to the above conditions.")
    st.write("**Author: [Sven Pfiffner](https://github.com/SvenPfiffner)**")
    if st.button("Okay"):
        st.rerun()

# Title of the app
st.title(":robot_face: SegmentBot")

# Sidebar settings
st.sidebar.title(":gear: Settings")
disclaimer = st.sidebar.button("Disclaimer", icon="⚠️")
if disclaimer:
    disclaimer_modal()
st.sidebar.markdown("---")
setting0 = st.sidebar.slider("Edit Strength", 0.0, 1.0, 0.5, 0.1, help="Some edit modes may have a strength parameter to control the intensity of the effect. Adjust it here. For modes without strength, this slider has no effect.")
setting1 = st.sidebar.text_input("Objects to mask", "", help="Enter the objects you want to mask, separated by commas")
setting2 = st.sidebar.text_input("Objects to keep", "", help="Enter the objects you explicitly dont want to be part of the mask, separated by commas.")
st.sidebar.markdown("---")
setting3 = st.sidebar.selectbox("Edit Mode", Editor.get_Subclasses().keys(), help="Select the edit mode you want to apply to the mask")
st.sidebar.text_area("Description", disabled=True, value=Editor.get_Subclasses()[setting3].getDescription())
st.sidebar.markdown("---")

st.sidebar.text("System Status")

# Display RAM usage
ram = psutil.virtual_memory()
st.sidebar.text(f"RAM Usage: {ram.percent}%")

# Display VRAM usage if CUDA is available
if torch.cuda.is_available():
    vram = torch.cuda.memory_allocated() / (1024 ** 3)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    st.sidebar.text(f"VRAM Usage: {vram:.2f} GB / {total_vram:.2f} GB")
    st.sidebar.success(":runner: CUDA available! Running on GPU")
else:
    st.sidebar.warning("No CUDA support detected! Running on CPU")

st.write("Welcome to SegmentBot! Effortlessly create masks and manipulate images with zero-shot segmentation technology. Upload an image, prompt objects of interest, and let the AI handle the rest.")
st.write("This project is based on the Open Source technology [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything). The application is designed to easily be extended with new edit styles and this web demo is meant as a showcase of some basic functionality. The full source code as well as detailed license attributions can be found on the [GitHub repository]()")


st.write("Upload an image from your device below and click the 'Process Image' button to start the process.")
# Upload image
uploaded_file = st.file_uploader("Choose an image to process", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    process_button = st.button(":camera: Process Image", use_container_width=True, type="primary")
    # Display the input image and processed image side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    with col2:
        if process_button:
            with st.spinner('Processing...'):
                # Here you can add the image processing code
                # For now, it just displays the prediction result
                image_array = np.array(Image.open(uploaded_file))

                settings = ProcessorSettings(strength=setting0,
                                            pos_prompt=setting1,
                                            neg_prompt=setting2,
                                            censorer_name=setting3
                                            )

                output_image = process_image(image_array, settings)
            
            # Display the output image
            st.image(output_image, caption='Processed Image.', use_column_width=True)
            
    if process_button:
        # Add a download button to store the output_image
        output_image_pil = Image.fromarray(output_image)
        output_image_pil.save("processed_image.png")
        with open("processed_image.png", "rb") as file:
            btn = st.download_button(
            label=":file_folder: Download Processed Image",
            data=file,
            file_name="processed_image.png",
            mime="image/png",
            use_container_width=True
        )
