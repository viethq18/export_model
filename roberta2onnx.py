import os
import torch
from transformers import AutoTokenizer, AutoModel

# PhoBERT

def convert_onnx(model, data_input, model_name = 'phobertv2.onnx', logs_dir = "./logs"):
    os.makedirs(logs_dir, exist_ok=True)
    torch.onnx.export(model,               # model being run
                    (data_input["input_ids"], data_input["token_type_ids"],  data_input["attention_mask"]), # model input (or a tuple for multiple inputs)
                    os.path.join(logs_dir, model_name),   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=13,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input_ids', 'token_type_ids', 'attention_mask'],   # the model's input names
                    output_names = ['logits'], # the model's output names
                    dynamic_axes={'input_ids' : {0 : 'batch_size', 1: 'sequence_len'},    # variable length axes
                                    'token_type_ids' : {0 : 'batch_size', 1: 'sequence_len'},
                                    'attention_mask' : {0 : 'batch_size', 1: 'sequence_len'},
                                    'logits' : {0 : 'batch_size'},
                                }
    )


if __name__ == '__main__':                    
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
    data_input = {
        "input_ids": torch.rand((3,512)).long(),
        "token_type_ids": torch.rand((3,512)).long(), 
        "attention_mask": torch.rand((3,512)).long()}
    
    model = AutoModel.from_pretrained("vinai/phobert-base-v2")
    model.to("cpu")
    model.eval()
    convert_onnx(model, data_input, logs_dir = "./")