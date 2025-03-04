!pip install unsloth
!pip install -U "huggingface_hub[cli]"

%%shell
huggingface-cli login --token Your_Token


import torch
from datasets import load_dataset
from trl import SFTTrainer



from unsloth import FastLanguageModel
model,tokenizer = FastLanguageModel.from_pretrained(
    model_name='unsloth/Llama-3.2-3B-Instruct-bnb-4bit',
     max_seq_length=2048,  #maximum number of tokens the model can process in a single input sequence
    load_in_4bit=True,
    dtype=None
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules = ["q_proj","o_proj","k_proj","gate_proj","up_proj","down_proj"],
    lora_alpha = 16,
 lora_dropout = 0, # Optimized at 0
 bias = "none", # No additional bias terms
 use_gradient_checkpointing = "unsloth", # Gradient checkpointing to save memory
 random_state = 3407,
 use_rslora = False
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules = ["q_proj","o_proj","k_proj","gate_proj","up_proj","down_proj"],
    lora_alpha = 16,
 lora_dropout = 0, # Optimized at 0
 bias = "none", # No additional bias terms
 use_gradient_checkpointing = "unsloth", # Gradient checkpointing to save memory
 random_state = 3407,
 use_rslora = False
)

dataset = load_dataset('opennyaiorg/aalap_instruction_dataset',split="train")
import pandas as pd
dataset=pd.DataFrame(dataset)
dataset.columns

df = dataset[dataset["task"].str.contains("argument_generation")]
df.head()

df["input_text"] = df["input_text"].str.replace(r'[\n"""[\]/]', '', regex=True)
df.head()
# dataset.head()
