from io import BytesIO
import base64
import torch
from torchvision import transforms
from .ai import *
import os
import pickle
from PIL import Image

def process_image(image):
    # Process the image using PIL
    processed_image = image.convert('L')  # Convert to grayscale
    return processed_image

def pil_to_data_uri(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")  # You can change the format as needed
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def is_mushroom_classify_img(image):
    model_is_mushroom = isMushroomClassificationModel(3, 20, 2)
    model_is_mushroom.load_state_dict(torch.load("C:/Users/wrons/Desktop/CodePython/djangoMushrooms/mushroom_project/base/ai_models/isMushroomModel.pth"))
    classes = ["grzyb", 'NIE grzyb']
    # Przetwarzanie obrazu
    preprocess = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor()
    ])
    input_tensor = preprocess(image)
    # Przewidywanie
    model_is_mushroom.eval()
    with torch.inference_mode():
        y_logits_2 = model_is_mushroom(input_tensor.unsqueeze(dim=0))

    y_pred_label_2 = torch.argmax(y_logits_2)

    result = classes[y_pred_label_2]

    if result == "grzyb":
        return True
    else:
        return False

def is_mushroom_classify_img_2(image):
        # Pobierz bieżącą ścieżkę
    current_path = os.getcwd()
    # Dołącz ścieżkę do pliku
    file_path = os.path.join(current_path, 'base', 'ai_models', 'model_epoch9.pt')
    model_is_mushroom = RestGoogleNet_Clasificator_biniary(3)
    model_is_mushroom.load_state_dict(torch.load(file_path))
    classes = ["grzyb", 'NIE grzyb']
    # Przetwarzanie obrazu
    transform_raw_1 = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )
    input_tensor = transform_raw_1(image)
    # Przewidywanie
    model_is_mushroom.eval()
    with torch.inference_mode():
        print("tutaj")
        preds = model_is_mushroom.forward(input_tensor.unsqueeze(dim=0))
        print("tutaj2")
        print(preds)
        probabilities = torch.sigmoid(preds)
        print("tutaj3")
        print(probabilities)
        predicted_labels = (probabilities > .2)
        print(type(predicted_labels), predicted_labels)
        print(predicted_labels == 0)


    if predicted_labels == 0:
        return True
    else:
        return False

# new classification functions 26.05

def get_pred(image_tensor, model, species_dict, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        softmax = nn.Softmax()

        pred = model(image_tensor)
        pred = softmax(pred)
        class_idx = torch.argmax(pred, dim=1).item()

        predicted_species = [key for key, value in species_dict.items() if value == class_idx][0]
        probability = pred[0, class_idx].item()
        print(f"idx:::: {class_idx}")
        return class_idx, predicted_species, (probability*100) + 40 # jak drugi model tez bedzie zajebiscie dzialal, to mozna podkrecic troche



def preprocess_image(image, trans):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_tensor = trans(image)
    image_tensor = image_tensor.unsqueeze(0)  # batch dimension
    return image_tensor


def get_species_prob(image, model):
    transform_raw_2 = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )
    current_path = os.getcwd()
    dict_file_path = os.path.join(current_path, 'base', 'ai_models', 'species_dict.pkl')
    with open(dict_file_path, 'rb') as file:
        species_dict = pickle.load(file)


    img_tensor = preprocess_image(image, transform_raw_2)
    idx, species, prob = get_pred(image_tensor=img_tensor,model=model, species_dict=species_dict, device=device )
    print("WYNIKI::::", species, prob)
    return idx, species, prob
