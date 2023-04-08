import streamlit as st
from annotated_text import annotated_text

import io
import os
import yaml
import pyarrow
import tokenizers


os.environ["TOKENIZERS_PARALLELISM"] = "true"

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

#@st.cache
def from_library():
    from retro_reader import RetroReader
    from retro_reader import constants as C
    return C, RetroReader

C, RetroReader = from_library()

# https://stackoverflow.com/questions/70274841/streamlit-unhashable-typeerror-when-i-use-st-cache
my_hash_func = {
    io.TextIOWrapper: lambda _: None,
    pyarrow.lib.Buffer: lambda _: 0,
    tokenizers.Tokenizer: lambda _: None,
    tokenizers.AddedToken: lambda _: None
}


def load_en_bert_large_model():
    config_file = "configs/inference_en_bert_large.yaml"
    return RetroReader.load(config_file=config_file)


def main():
    option = st.selectbox(
        label="Please choose the language you want to use",
        options=(
            "English",
            "Vietnamese"
        ),
        index=0,
    )
    
    if option == "English":
        st.title("Retrospective Reader Demo")
        retro_reader = load_en_bert_large_model()
        lang_prefix = "EN"
        height = 300
        
        
        
        st.markdown("## Demonstration")
        with st.form(key="my_form"):
            query = st.text_input(
                label="Type your query",
                max_chars=None,
                help=getattr(C, f"{lang_prefix}_QUERY_HELP_TEXT"),
            )
            context = st.text_area(
                label="Type your context",
                height=height,
                max_chars=None,
                help=getattr(C, f"{lang_prefix}_CONTEXT_HELP_TEXT"),
            )
            submit_button = st.form_submit_button(label="Submit")
            return_submodule_outputs = st.checkbox('return_5_outputs', value=False)
        
        if submit_button:
            with st.spinner("Please wait a little bit.."):
                outputs = retro_reader(
                    query=query,
                    context=context,
                    return_submodule_outputs=return_submodule_outputs,
                )
            answer = outputs[0]["id-01"]
            nbest_preds = outputs[1]
            highest_prob = outputs[2]
            if not answer:
                answer = "No answer"
            if not return_submodule_outputs:
                st.markdown("## Highest possible answer is")
                st.write(answer)
                ans_tuple = (answer, "", "#faa")
                annotated_text(ans_tuple)
                st.markdown("## Probability for this answer is")
                st.write(highest_prob)
            else:
                st.markdown("## 5 highest possible answers are")
                st.json(nbest_preds)
             
        
    else:
        st.title("Demo Bộ đọc Hồi tưởng")
    
    
    
   
    
    
    
    
    
    
    
    
        
    
        
        
        

       
        
        

        
    

if __name__ == "__main__":
    main()
