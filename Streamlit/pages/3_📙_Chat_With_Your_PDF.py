import streamlit as st
import streamlit.components.v1 as components

# embed streamlit docs in a streamlit app
components.iframe("http://10.254.213.127:5174/", height=650, width=900, scrolling=False)

st.markdown(
    """
    <style>
        .block-container {
            margin-left: -230px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

html_code = """
<div style='text-align: center;'>
    <p style="
        font-size: 0.8rem;
        color: rgb(107, 114, 128);
        width: 100%;
        height: 100%;
        position: relative;
        margin: 0;
        padding:0.5rem 0;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        background: #fff;
        border-top: 1px solid #eee;
    ">
        Powered by <a href="https://github.com/intel/intel-extension-for-transformers" style="text-decoration: underline; margin-left:5px;" target="_blank"> Intel Extension for Transformers</a>
    </p>
</div>

<style>
    .line::before {
        left: 0;
        margin-left: -10px;
    }

    .line::after {
        right: 0;
        margin-right: -10px;
    }
</style>
"""

st.markdown(html_code, unsafe_allow_html=True)