from imports import *
from img_generation import *

st.title("Unconditional Image Generation")

if st.button('Generate Image'):

    with st.spinner('Generating images. . .'):
        generated_images = gen_img()
        cols = st.columns(4)
        for i in range(3):
            for j in range(4):
                with cols[j]:
                    st.image(generated_images[i * 4 + j], caption=f"Generated Image {i * 4 + j + 1}")

    st.markdown('<div class="center"><div class="image-container">', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)