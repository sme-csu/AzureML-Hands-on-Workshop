import json
import sys
import os
import time
from azureml.core.model import Model
import numpy as np
#import onnxruntime

def init():
    global session
    model=os.path.join(os.getenv('AZUREML_MODEL_DIR'),'cifar10_net.onnx')
    session=onnxruntime.InferenceSession(model)
    # input=session.get_inputs()[0].name
    # output=session.get_outputs()[0].name



def preprocess(input_data_json):
    return np.array(json.loads(input_data_json)['data']).astype('float32')

# def postprocess(result):
#     return int(np.argmax(np.array(result).squeeze(), axis=0))


def run(input_data_json):
    try:
        start=time.time()
        input_data=preprocess(input_data_json)
        input_name=session.get_inputs()[0].name
        result=session.run([],{input_name:input_data})
        end=time.time()
        prediction=category_map(postprocess(result[0]))

        result.dict= {"result" : postprocess(prediction)}
        return 
    except Exception as e:
        result=str(e)
        return {"error":result}

def category_map(classes):
    category_map={"airplane":0, "automobile": 1, "bird": 2, "cat": 3, "deer" :4, "dog": 5,
                 "frog": 6,"horse": 7, "ship": 8, "truck": 9}
    
    category_key=list(category_map.keys())

    category=[]
    category.append(category_key[classes[0]])
    return category

def softmax(x):
    x=x.reshape(-1)
    e_x=np.exp(x-np.max(x))
    return e_x/e_x.sum(axis=0)

def postprocess(scores):
    prob=softmax(scores)
    prob=np.squeeze(prob)
    classes=np.argsort(prob)[::-1]
    return classes
























# def preprocess(input_data_json):
#     # convert the JSON data into the tensor input
#         transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])
#     input_data=transform(input_data_json)

#     # return np.array(json.loads(input_data_json)['data']).astype('float32')
#     return input_data

# def postprocess(result):
#     # We use argmax to pick the highest confidence label
#     return int(np.argmax(np.array(result).squeeze(), axis=0))

# def run(input_data_json):
#     try:
        
#         input_data = preprocess(input_data_json)
#         input_name = session.get_inputs()[0].name  # get the id of the first input of the model   
#         result = session.run([], {input_name: input_data})
#         return {"result": postprocess(result)}
#     except Exception as e:
#         result = str(e)
#         return {"error": result}



# # def run(input_data):
# #     try:
# #         data=np.array(json.loads(input_data)['data']).astype('float32')

# #         r=session.run([output],{input :data})

# #         result=index_map(postprocess(r[0]))

# #         result_dict={"result": result}
# #     except Exception as e:
# #         result_dict={"error": str(e)}
    
# #     return json.dumps(result_dict)


# # def index_map(classes):
# #     index_map={"airplane":0, "automobile": 1, "bird": 2, "cat": 3, "deer" :4, "dog": 5, "frog": 6,"horse": 7, "ship": 8, "truck": 9}
# #     index_key=list(index_map.keys())

# #     index=[]
# #     index.append(index_key[classes[i]])
# #     return index



# # def softmax(x):
# #     x=x.reshape(-1)
# #     e_x=np.exp(x-np.max(x))
# #     return e_x/e_x.sum(axis=0)

# # def postprocess(scores):
# #     prob=softmax(scores)
# #     prob=np.squeeze(prob,axis=0)
# #     classes=np.argsort(prob)[::-1]
# #     return classes