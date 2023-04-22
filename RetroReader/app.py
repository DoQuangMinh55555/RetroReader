import streamlit as st
from annotated_text import annotated_text

import io
import os
import yaml
import pyarrow
import tokenizers
import collections


os.environ["TOKENIZERS_PARALLELISM"] = "true"

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

#@st.cache
def from_library():
    from retro_reader import RetroReader
    from retro_reader import constants as C
    return C, RetroReader

C, RetroReader = from_library()


my_hash_func = {
    io.TextIOWrapper: lambda _: None,
    pyarrow.lib.Buffer: lambda _: 0,
    tokenizers.Tokenizer: lambda _: None,
    tokenizers.AddedToken: lambda _: None
}


def load_en_bert_large_model():
    config_file = "configs/inference_en_bert_large.yaml"
    return RetroReader.load(config_file=config_file)

# def load_vi_bert_large_model():
#     config_file = "configs/inference_vi_bert_large.yaml"
#     return RetroReader.load(config_file=config_file)

def format_context(context: str, answer_start: int, answer_end: int):
    context_before_answer = context[:answer_start]
    context_answer = context[answer_start:answer_end]
    context_after_answer = context[answer_end:]
    context_formatted = annotated_text(context_before_answer, (context_answer, "", "#ff0"), context_after_answer)
    return context_formatted


def main():
    option = st.selectbox(
        label="Please choose the language you want to use/Vui lòng chọn ngôn ngữ mà bạn mong muốn sử dụng",
        options=(
            "English/Tiếng Anh",
            "Vietnamese/Tiếng Việt"
        ),
        index=0,
    )
    
    if option == "English/Tiếng Anh":
        st.title("Retrospective Reader Demo")
        retro_reader = load_en_bert_large_model()
        lang_prefix = "EN"
        height = 300
        
        
        
        st.markdown("## Demonstration")
        with st.form(key="my_form"):
            query = st.text_input(
                label="Please input your query",
                max_chars=None,
                help=getattr(C, f"{lang_prefix}_QUERY_HELP_TEXT"),
            )
            context = st.text_area(
                label="Please input your context",
                height=height,
                max_chars=None,
                help=getattr(C, f"{lang_prefix}_CONTEXT_HELP_TEXT"),
            )
            submit_button = st.form_submit_button(label="Answer!")
            return_submodule_outputs = st.checkbox('Return 5 highest possible outputs', value=False)
        
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
            answer_start = outputs[3]
            if not answer:
                answer = "No answer"
                answer_start = -1
            if not return_submodule_outputs:
                st.markdown("## Highest possible answer is")
                st.write(answer)
                #ans_tuple = (answer, "", "#faa")
                #annotated_text(ans_tuple)
                st.markdown("## Probability for this answer is")
                st.write(highest_prob)
                #st.markdown("## Answer start is")
                #st.write(answer_start)
                if answer_start != -1:
                    answer_end = answer_start + len(answer)
                    #st.write(format_context(context, answer_start, answer_end))
                    context = format_context(context, answer_start, answer_end)
            else:
                st.markdown("## 5 highest possible answers are")
                best_preds = collections.OrderedDict()
                best_preds["id-01"] = []
                for i in range(5):
                    best_preds["id-01"].append(
                        {
                            "text": nbest_preds["id-01"][i]["text"],
                            "probability": nbest_preds["id-01"][i]["probability"],
                        }
                    )
                #st.json(nbest_preds)
                st.json(best_preds)
             
        
    else:
        st.title("Demo Bộ đọc Hồi tưởng")
        #retro_reader = load_vi_bert_large_model()
        lang_prefix = "EN"
        height = 300
        
        st.markdown("## Hiện thực hệ thống")
        with st.form(key="my_form"):
            query = st.text_input(
                label="Vui lòng nhập câu truy vấn của bạn",
                max_chars=None,
                help=getattr(C, f"{lang_prefix}_QUERY_HELP_TEXT"),
            )
            context = st.text_area(
                label="Vui lòng cung cấp ngữ cảnh",
                height=height,
                max_chars=None,
                help=getattr(C, f"{lang_prefix}_CONTEXT_HELP_TEXT"),
            )
            submit_button = st.form_submit_button(label="Trả lời câu hỏi!")
            return_submodule_outputs = st.checkbox('Trả về 5 câu trả lời khả thi nhất', value=False)
#         if submit_button:
#             with st.spinner("Vui lòng chờ trong giây lát.."):
#                 outputs = retro_reader(
#                     query=query,
#                     context=context,
#                     return_submodule_outputs=return_submodule_outputs,
#                 )
#             answer = outputs[0]["id-01"]
#             nbest_preds = outputs[1]
#             highest_prob = outputs[2]
#             answer_start = outputs[3]
#             if not answer:
#                 answer = "Không tìm được câu trả lời"
#                 answer_start = -1
#             if not return_submodule_outputs:
#                 st.markdown("## Câu trả lời có khả năng cao nhất là")
#                 st.write(answer)
#                 #ans_tuple = (answer, "", "#faa")
#                 #annotated_text(ans_tuple)
#                 st.markdown("## Xác suất cho câu trả lời này là")
#                 st.write(highest_prob)
#                 #st.markdown("## Answer start is")
#                 #st.write(answer_start)
#                 if answer_start != -1:
#                     answer_end = answer_start + len(answer)
#                     #st.write(format_context(context, answer_start, answer_end))
#                     context = format_context(context, answer_start, answer_end)
#             else:
#                 st.markdown("## 5 câu trả lời khả thi nhất là")
#                 st.json(nbest_preds)
        
    
    
    
   
    
    
    
    
    
    
    
    
        
    
        
        
        

       
        
        

        
    

if __name__ == "__main__":
    main()
