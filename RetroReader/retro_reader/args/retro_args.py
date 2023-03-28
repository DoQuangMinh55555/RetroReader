from dataclasses import dataclass, field
from .. import models


@dataclass
class RetroDataModelArguments:
    pass


@dataclass
class DataArguments(RetroDataModelArguments):
    max_seq_length: int = field(
        default=512,
        metadata={"help": ""},
    )
    max_answer_length: int = field(
        default=30,
        metadata={"help": ""},
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": ""},
    )
    return_token_type_ids: bool = field(
        default=True,
        metadata={"help": ""},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={"help": ""},
    )
    preprocessing_num_workers: int = field(
        default=5,
        metadata={"help": ""},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": ""},
    )
    version_2_with_negative: bool = field(
        default=True,
        metadata={"help": ""},
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={"help": ""},
    )
    rear_threshold: float = field(
        default=0.0,
        metadata={"help": ""},
    )
    n_best_size: int = field(
        default=5,
        metadata={"help": ""},
    )
    use_choice_logits: bool = field(
        default=False,
        metadata={"help": ""},
    )
    start_n_top: int = field(
        default=-1,
        metadata={"help": ""},
    )
    end_n_top: int = field(
        default=-1,
        metadata={"help": ""},
    )
    beta1: int = field(
        default=1,
        metadata={"help": ""},
    )
    beta2: int = field(
        default=1,
        metadata={"help": ""},
    )
    best_cof: int = field(
        default=1,
        metadata={"help": ""},
    )
    
    
@dataclass
class ModelArguments(RetroDataModelArguments):
    use_auth_token: bool = field(
        default=False,
        metadata={"help": ""},
    )
        
        
@dataclass
class SketchModelArguments(ModelArguments):
    # sketch_revision: str = field(
    #     #default="en-bert-large-sketch",
    #     #default=None,
    #     default="main",
    #     metadata={"help": ""},
    # )
    sketch_model_name: str = field(
        #default="bert-large-uncased",
        #default="quangminhdo/retro-reader-for-LV",
        default="quangminhdo/retro-reader-for-LV-sketch",
        metadata={"help": ""},
    )
    sketch_tokenizer_name: str = field(
        #default="quangminhdo/retro-reader-for-LV/sketch",
        default=None,
        metadata={"help": ""},
    )
    sketch_architectures: str = field(
        #default="BertForSequenceClassification",
        #default=None,
        default="BertForSequenceClassification",
        metadata={"help": ""},
    )
            
            
@dataclass
class IntensiveModelArguments(ModelArguments):
    # intensive_revision: str = field(
    #     #default="en-bert-large-intensive",
    #     #default=None,
    #     default="main",
    #     metadata={"help": ""},
    # )
    intensive_model_name: str = field(
        #default="bert-large-uncased",
        #default="quangminhdo/retro-reader-for-LV",
        default="quangminhdo/retro-reader-for-LV-intensive",
        metadata={"help": ""},
    )
    intensive_tokenizer_name: str = field(
        #default="quangminhdo/retro-reader-for-LV/intensive",
        default=None,
        metadata={"help": ""},
    )
    intensive_architectures: str = field(
        #default="BertForQuestionAnsweringAVPool",
        default="BertForQuestionAnsweringAVPool",
        metadata={"help": ""},
    )
    
    
@dataclass
class RetroArguments(
    DataArguments, 
    SketchModelArguments, 
    IntensiveModelArguments,
):
    def __post_init__(self):
        # Sketch
        model_cls = getattr(models, self.sketch_architectures, None)
        if model_cls is None:
            raise AttributeError
        self.sketch_model_cls = model_cls
        self.sketch_model_type = model_cls.model_type
        if self.sketch_tokenizer_name is None:
            self.sketch_tokenizer_name = self.sketch_model_name
        # Intensive
        model_cls = getattr(models, self.intensive_architectures, None)
        if model_cls is None:
            raise AttributeError
        self.intensive_model_cls = model_cls
        self.intensive_model_type = model_cls.model_type
        if self.intensive_tokenizer_name is None:
            self.intensive_tokenizer_name = self.intensive_model_name