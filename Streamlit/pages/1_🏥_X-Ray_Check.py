import streamlit as st
import re
from tempfile import NamedTemporaryFile
import visual_med_alpaca as med

MAX_QUESTION_LEN = 256

st.set_page_config(
    page_title="XRay Check Demo",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: black;
    color: white;
}
</style>""", unsafe_allow_html=True)

if 'result' not in st.session_state:
    st.session_state['result'] = ""

def filter_response(result):
    print(f"result before filter: {result}")
    regex = r"### Response:\n.*?###"
    match = re.findall(regex, result, re.DOTALL) # DOTALL make '.' matching with newline also
    if len(match) > 0:
        response = match[0][len("### Response:\n"):-3]
    else:
        # Try pattern match again
        regex = r"### Response:\n.*"
        match = re.findall(regex, result, re.DOTALL)
        if len(match) > 0:
            response = match[0][len("### Response:\n"):]
        else: # don't filter if no any match
            response = result
    return response

def handle_submit():
    st.session_state['result'] = ""
    if img is None and (question is None or question == ""):
        box.error("You need input question or/and upload image. You can't leave them both unfilled!")
        return

    # Question only
    if img is None and question != "":
        with st.status("Generating resposne...", state="running"):
            st.session_state['result'] = filter_response(med.infer_alpaca(question, ""))
        print(f"result: {st.session_state['result']}")
        return

    # Img w/wo Question
    assert img is not None
    with NamedTemporaryFile(dir='.') as f:
        f.write(img.getbuffer())
        f.flush()
        with st.status("Generating resposne...", state="running"):
            caption = med.infer_git(f.name)
            assert caption != ""
            if question is None or question == "":
                # Img only
                st.session_state['result'] = caption
                print(f"result: {st.session_state['result']}")
                return
            # Img + Question
            st.session_state['result'] = filter_response(med.infer_alpaca(question, caption))

    print(f"result: {st.session_state['result']}")
    return

st.header("🏥 X-Ray Image Check Demo")
st.divider()

box = st.container(border=True)

box.markdown("<h3 style='text-align: center; color: black;'>Visual Med-Alpaca: Bridging Modalities in Biomedical Language Models</h3>", unsafe_allow_html=True)
box.markdown("<h6 style='text-align: center; color: gray;'>To use this demo, simply upload your image and type a question or instruction and click 'submit'</h6>", unsafe_allow_html=True)

box.text("")

lc, rc = box.columns((3,2), gap="large")

# File uploader area
file_lc, file_rc = lc.container().columns((1,5))
file_lc.markdown("**Input Image**")
img = file_rc.file_uploader("Select an image", type=['.jpg'], label_visibility="collapsed")
if img is not None:
    file_rc.image(img, width=350)
lc.checkbox("Use finetuned model?", value=False, key="git-finetuned")
lc.divider()

# Question input area
lc.markdown("**Question/instruction**")
question = lc.text_input("Question", value="", max_chars=MAX_QUESTION_LEN, label_visibility="collapsed")
# lc.divider()
lc.text("")

# Model selection area
#lc.markdown("Choose your LLM")
#model = lc.selectbox("LLM", options=['med-alpaca', 'med-alpaca-lora'], label_visibility="collapsed")
lc.radio("Choose your LLM:", options=['Alpaca base', 'Alpaca FT', 'Alpaca LORA FT'], index=0, key='LLM-model')

lc.button("Submit", use_container_width=True, on_click=handle_submit)

# Right column
rc.divider()
rc.markdown("**Output**")
output_area = rc.container(height=300, border=True)
output_area.write(st.session_state['result'])
