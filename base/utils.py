from io import BytesIO
import base64
import torch
from torchvision import transforms
from .ai import isMushroomClassificationModel

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