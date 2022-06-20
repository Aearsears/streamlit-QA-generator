import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from annotated_text import annotated_text

# To interate through results
from collections import Counter
from pipelines import pipeline
import nltk
import requests
import json

nltk.download("popular")


query_params = st.experimental_get_query_params()
# region Layout size

st.set_page_config(page_title="Q&A Generator", page_icon="🎈")


def _max_width_():
    max_width_str = f"max-width: 1700px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()

st.title("Q&A Generator")

st.write(
    "This Q&A Generator leverages the power of [Google T5 Transformer](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) to generate quality question/answer pairs from content.")

st.header("")

c3, c4, c5 = st.columns([1, 6, 1])

with c4:
    with st.form("Form"):
        valuetxt = query_params["text"][0] if "text" in query_params else ""
        URLBox = st.text_area("👇 Paste text below to get started!",
                              placeholder="Ex. Elon Musk is the CEO of Tesla", help="Don't put more than 1000 words", value=valuetxt)
        cap = 1000

        submitted = st.form_submit_button("Get your Q&A pairs")
        if valuetxt != "":
            submitted = True

    c = st.container()

    if not submitted and not URLBox:
        st.stop()

    if submitted and not URLBox:
        st.warning("☝️ Please add some text.")
        st.stop()

text2 = (URLBox[:cap] + ".") if len(URLBox) > cap else URLBox
lenText = len(text2)

if lenText > cap:
    c.warning(
        "We will build the Q&A pairs based on the first 1,000 characters."
    )
else:
    pass

try:
    nlp = pipeline("multitask-qa-qg")
    faqs = nlp(text2)
    st.json(faqs)
    url = 'https://cardify-backend.herokuapp.com/qareceive'
    x = requests.post(url, json=json.dumps(faqs))
    st.title("Status Code")

except Exception as e:
    st.warning(
        f"""
    🔮 **Snap!** Seems like there's an issue with that block of text, please try another one. 
    """
    )
    st.stop()
