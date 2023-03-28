import streamlit as st

import io
import os
import yaml
import pyarrow
import tokenizers


os.environ["TOKENIZERS_PARALLELISM"] = "true"

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

@st.cache
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

# @st.cache(hash_funcs=my_hash_func, allow_output_mutation=True)
# def load_ko_roberta_large_model():
#     config_file = "configs/inference_ko_roberta_large.yaml"
#     return RetroReader.load(config_file=config_file)


# @st.cache(hash_funcs=my_hash_func, allow_output_mutation=True)
# def load_ko_electra_small_model():
#     config_file = "configs/inference_ko_electra_small.yaml"
#     return RetroReader.load(config_file=config_file)


# @st.cache(hash_funcs=my_hash_func, allow_output_mutation=True)
# def load_en_electra_large_model():
#     config_file = "configs/inference_en_electra_large.yaml"
#     return RetroReader.load(config_file=config_file)

# @st.cache(hash_funcs=my_hash_func, allow_output_mutation=True)
# def load_en_albert_xxlarge_model():
#     config_file = "configs/train_en_albert_xxlarge.yaml"
#     return RetroReader.load(config_file=config_file)

@st.cache(hash_funcs=my_hash_func, allow_output_mutation=True)
def load_en_bert_large_model():
    config_file = "configs/inference_en_bert_large.yaml"
    return RetroReader.load(config_file=config_file)


# RETRO_READER_HOST = {
#     # "klue/roberta-large": load_ko_roberta_large_model(),
#     #"monologg/koelectra-small-v3-discriminator": load_ko_electra_small_model(),
#     #"google/electra-large-discriminator": load_en_electra_large_model(),
#     #"albert-xxlarge-v2": load_en_albert_xxlarge_model(), 
#     "bert-large-uncased": load_en_bert_large_model()
# }


def main():
    st.title("Retrospective Reader Demo")
    
    # st.markdown("## Model name")
    # option = st.selectbox(
    #     label="Choose the model used in retro reader",
    #     options=(
    #         "bert-large-uncased"
    #     ),
    #     index=0,
    # )
    # model_name = option
    
    # retro_reader = RETRO_READER_HOST[model_name]
    
    retro_reader = load_en_bert_large_model()
    lang_prefix = "EN"
    height = 300
    
    retro_reader.null_score_diff_threshold = st.sidebar.slider(
        label="null_score_diff_threshold",
        min_value=-10.0, max_value=10.0, value=0.0, step=1.0,
        help="ma!",
    )
    retro_reader.rear_threshold = st.sidebar.slider(
        label="rear_threshold",
        min_value=-10.0, max_value=10.0, value=0.0, step=1.0,
        help="ma!",
    )
    retro_reader.n_best_size = st.sidebar.slider(
        label="n_best_size",
        min_value=1, max_value=20, value=5, step=1,
        help="ma!",
    )
    retro_reader.beta1 = st.sidebar.slider(
        label="beta1",
        min_value=-10.0, max_value=10.0, value=1.0, step=1.0,
        help="ma!",
    )
    retro_reader.beta2 = st.sidebar.slider(
        label="beta2",
        min_value=-10.0, max_value=10.0, value=1.0, step=1.0,
        help="ma!",
    )
    retro_reader.best_cof = st.sidebar.slider(
        label="best_cof",
        min_value=-10.0, max_value=10.0, value=1.0, step=1.0,
        help="ma!",
    )
    return_submodule_outputs = st.sidebar.checkbox('return_submodule_outputs', value=False)
    
    st.markdown("## Demonstration")
    with st.form(key="my_form"):
        query = st.text_input(
            label="Type your query",
            value=getattr(C, f"{lang_prefix}_EXAMPLE_QUERY"),
            max_chars=None,
            help=getattr(C, f"{lang_prefix}_QUERY_HELP_TEXT"),
        )
        context = st.text_area(
            label="Type your context",
            value=getattr(C, f"{lang_prefix}_EXAMPLE_CONTEXTS"),
            height=height,
            max_chars=None,
            help=getattr(C, f"{lang_prefix}_CONTEXT_HELP_TEXT"),
        )
        submit_button = st.form_submit_button(label="Submit")
        
    if submit_button:
        with st.spinner("Please wait.."):
            outputs = retro_reader(
                query=query,
                context=context,
                return_submodule_outputs=return_submodule_outputs,
            )
        #answer, score, nbest_preds = outputs[0]["id-01"], outputs[1], outputs[2]
        answer = outputs[0]["id-01"]
        nbest_preds = outputs[1]
        #highest_prob = str(outputs[1]["id-01"][0]["probability"])
        highest_prob = outputs[2]

        if not answer:
            answer = "No answer"
        st.markdown("## The answer is")
        st.write(answer)
        st.markdown("### Probability for this answer is")
        st.write(highest_prob)
        # st.markdown("### Rear Verification Score")
        # st.json(score)

        if return_submodule_outputs:
            # st.markdown("### Sketch Reader Score (score_ext)")
            # st.json(score_ext)
            # st.markdown("### Intensive Reader Score (score_diff)")
            # st.json(score_diff)
            st.markdown("### N Best Predictions (from intensive reader)")
            st.json(nbest_preds)
    

if __name__ == "__main__":
    main()