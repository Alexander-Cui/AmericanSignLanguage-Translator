import torch
from torchvision import models, transforms
import torch.nn as nn
import cv2
from PIL import Image

def initialize_model(num_classes):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features 
    model_ft.fc = nn.Linear(num_ftrs, num_classes) #match final layer to num_classes
    input_size = 224

    return model_ft, input_size

def transform_img (img_array, input_size=224): #if img size isnt given, default to 224

        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) #convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        my_transform = transforms.Compose([transforms.Resize(input_size), transforms.ToTensor()]) #convert to tensor
        my_img = my_transform(img)
        my_img = my_img.unsqueeze(0)
        return my_img
        
def predict(model, img_array):
        tensor = transform_img(img_array)
        outputs = model(tensor)
        _, pred = torch.max(outputs,1)

        return pred.item()


num_classes = 24
PATH = R'D:\Github\MAIS202\AmericanSignLanguageTranslator\Model_Weights\Resnet-Feature_extract_False_gray_new_data.pth'

model, input_size = initialize_model(num_classes)
model.load_state_dict(torch.load(PATH))
model.eval()