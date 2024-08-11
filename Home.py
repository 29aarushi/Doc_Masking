import streamlit as st
import graphviz

# Set page configuration
st.set_page_config(
    page_title="InterviewX",
    page_icon="🪪",
    layout="wide",
)

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



# Main page function
def main_page():
    Overview_col, Img_col = st.columns(spec=(1, 1.2), gap="large")

    with Overview_col:
        # Content for main page
        st.markdown(
            "<h1 style='text-align: left; font-size: 65px; '>Personal Doc Masking</h1>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            "<p style='font-size: 22px; text-align: left;'>Our aim is to provide a seamless solution for identifying personal documents and ensuring their confidentiality through effective masking techniques. In today's digital age, safeguarding personal information is paramount, and our application is designed to assist individuals and organizations in achieving this goal.</p>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div>
            <ul>
                <li><p style='font-size: 22px; text-align: left;'><strong>Avoid scams effortlessly!</strong> Privacy Protection: Personal document masking and detection techniques help safeguard sensitive information by obscuring or redacting personally identifiable information (PII) such as names, addresses, and identification numbers.</p></li>
                <li><p style='font-size: 22px; text-align: left;'><strong>Personal document masking and detection can also play a vital role in fraud prevention and security. By accurately identifying and verifying personal documents such as IDs, passports, and driver's licenses, businesses and institutions can authenticate the identity of individuals during various processes.</p></li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.write("")

        # Buttons for giving the profile links of the team members
        st.markdown(
            "<h3 style='text-align: left;'>Team members</h3>",
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            social_col1, social_col2, social_col3 = st.columns(
                spec=(1, 1, 1), gap="large"
            )
            with social_col1:
                st.link_button(
                    "Yuvraj Singh",
                    use_container_width=True,
                    url="https://github.com/yuvraaj2002",
                )

            with social_col2:
                st.link_button(
                    "Aarushi",
                    use_container_width=True,
                    url="https://www.linkedin.com/in/yuvraj-singh-a4430a215/",
                )

            with social_col3:
                st.link_button(
                    "KirtPreet",
                    use_container_width=True,
                    url="https://twitter.com/Singh_yuvraaj1",
                )

       

    with Img_col:
        st.markdown("<div class='top-margin'> </div>", unsafe_allow_html=True)
        st.image(r"artifacts\Banner.jpg")


        


main_page()
