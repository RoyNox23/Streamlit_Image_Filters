import streamlit as st
import numpy     as np
import base64
from PIL         import Image
from io          import BytesIO
from RNB_Kernel  import Kernels
from RNB_Filters import Filtering


def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href


st.title("Web Image Filtering App")
img_file_buffer = st.file_uploader("Upload an image.", type = ['jpg', 'jpeg', 'png'])
filter_applier  = Filtering(img_file_buffer, "BGR")
kernel          = Kernels()

if filter_applier.get_buffer() is not None:

    columns = st.columns(2)

    columns[0].image(filter_applier.get_cv2_image(), channels = "BGR")
    columns[0].text("Uploaded Image")

    filters_list = np.array(["None",
                             "Black&White",
                             "Red Channel",
                             "Green Channel",
                             "Blue Channel",
                             "Equalization",
                             "Linear Blur",
                             "Gaussian Blur",
                             "Median Blur",
                             "Bilateral Blur",
                             "Sharpened",
                             "Sobel",
                             "Canny"])

    selected_filter = st.selectbox("Select Filter", filters_list)

    kernel.set_kernel(selected_filter)

    filter_applier.applyFilter(selected_filter, kernel)

    if (filter_applier.get_filtered())[1] == "BGR":
        columns[1].image(filter_applier.get_filtered()[0], channels = "BGR")
    else:
        columns[1].image(filter_applier.get_filtered()[0])
    columns[1].text("Filtered Image")

    dl_image = Image.fromarray(filter_applier.get_filtered()[0][:, :, ::-1])
    st.markdown(get_image_download_link(dl_image,
                                        "My_filtered_image.jpg",
                                        'Download Output Image'),
                unsafe_allow_html = True)

