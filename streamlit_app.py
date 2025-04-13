import streamlit as st
from transformers import pipeline
import pandas as pd

# Set up the page
st.set_page_config(page_title="NER Demo", layout="wide")

# Title and description
st.title("🧠 Named Entity Recognition (NER) Demo")
st.markdown("""
This app uses a pre-trained BERT model to identify named entities in your text.  
Entities include **persons**, **locations**, **organizations**, and more.
""")

# Cache the NER pipeline
@st.cache_resource
def load_ner_pipeline():
    return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

ner = load_ner_pipeline()

# Text input
text = st.text_area("✍️ Enter your text below:", "Narendra Modi was born in Gujarat and served as prime minister of the India.")

if st.button("🔍 Analyze Entities"):
    with st.spinner("Analyzing..."):
        results = ner(text)

        if results:
            st.success("Entities detected:")
            df = pd.DataFrame(results)
            df = df[["word", "entity_group", "score", "start", "end"]]
            df.columns = ["Entity", "Label", "Confidence", "Start", "End"]
            df["Confidence"] = df["Confidence"].apply(lambda x: f"{x*100:.2f}%")
            st.dataframe(df)

            # Highlight in context
            st.markdown("### 📌 Highlighted Entities in Text")
            highlighted_text = text
            offset = 0
            for r in sorted(results, key=lambda x: x['start']):
                start = r['start'] + offset
                end = r['end'] + offset
                tag = f"<mark style='background-color:#ffeeba; border-radius:3px;'>{text[start:end]} <sub>[{r['entity_group']}]</sub></mark>"
                highlighted_text = highlighted_text[:start] + tag + highlighted_text[end:]
                offset += len(tag) - (end - start)

            st.markdown(f"<div style='line-height:1.8'>{highlighted_text}</div>", unsafe_allow_html=True)
        else:
            st.warning("No entities were detected.")
