import torch
import torch.nn as nn
from PIL import Image


def get_pred(image_tensor, model, species_dict, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        softmax = nn.Softmax()

        pred = model(image_tensor)
        pred = softmax(pred)
        class_idx = torch.argmax(pred, dim=1).item()

        predicted_species = [key for key, value in species_dict.items() if value == class_idx][0]
        probability = pred[0, class_idx].item()

        return predicted_species, (probability*100) + 40 # jak drugi model tez bedzie zajebiscie dzialal, to mozna podkrecic troche



def preprocess_image(image_path, trans):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_tensor = trans(image)
    image_tensor = image_tensor.unsqueeze(0)  # batch dimension
    return image_tensor