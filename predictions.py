import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, datasets
from model import isMushroomClassificationModel
from matplotlib import pyplot as plt
from pathlib import Path

model = isMushroomClassificationModel(3, 20, 2)
model.load_state_dict(torch.load("models/model_0.pth"))

classes = ["grzyb", 'NIE grzyb']

# image = Image.open("grzyb-panienka.jpg")
image = Image.open("em.jpg")

preprocess = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor()
])

input_tensor = preprocess(image)
print(input_tensor.shape)
print(input_tensor)

# Przewidywanie
model.eval()
with torch.inference_mode():
    y_logits_2 = model(input_tensor.unsqueeze(dim=0))

y_pred_label_2 = torch.argmax(y_logits_2)
result = classes[y_pred_label_2]
print(result)

# numpy_image = input_tensor.permute(1, 2, 0).numpy()  # Permutujemy wymiary i konwertujemy na NumPy

# # Wyświetlenie obrazu za pomocą matplotlib
# plt.imshow(numpy_image)
# plt.axis('off')  # Wyłączenie osi
# plt.show()