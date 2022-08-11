from transformers import pipeline
import torch

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    device = 0 if torch.cuda.is_available() else -1
    print("Active Device:{}".format(device))
    model = pipeline('question-answering', model='deepset/xlm-roberta-large-squad2', device=device)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    context = model_inputs.get('context', None)
    question = model_inputs.get('question', None)
    if context == None:
        return {'message': "No context provided"}
    if question == None:
        return {'message': "No question provided"}
    
    # Run the model
    result = model({'context': context,'question': question})

    # Return the results as a dictionary
    return result
