import os
import torch
import torch.nn as nn
from torchvision import transforms
import json

import torch.nn.functional as F



def init():
    global model
    
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'cifar10_net.pt')
    # model=torch.load(model_path)
    # model.load_state_dict(torch.load(model_path)
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.eval()


def run(input_data):
    input_data = torch.tensor(json.loads(input_data)['data'])

    # get prediction
    with torch.no_grad():
        output = model(input_data)
        classes = ['airplane', 'automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        softmax = nn.Softmax(dim=1)
        pred_probs = softmax(output).numpy()[0]
        index = torch.argmax(output, 1)

    result = {"label": classes[index], "probability": str(pred_probs[index])}
    return result