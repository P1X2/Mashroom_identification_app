import torch
from torchvision import transforms
import pickle

from Net import RestGoogleNet_Clasificator
from help_fnc import get_pred, preprocess_image

device = "cuda"
transform_raw_2 = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
)
with open('species_dict.pkl', 'rb') as file:
    species_dict = pickle.load(file)


####################################################3

num_classes = 17
model = RestGoogleNet_Clasificator(in_channels=3, num_classes=num_classes)
model.to(device)
model.load_state_dict(torch.load('models_SPECIES\e59_model_Species.pt'))

#####################################################

img_tensor = preprocess_image("grzybki\grzybek5.jpg", transform_raw_2)
species, prob = get_pred(image_tensor=img_tensor,model=model, species_dict=species_dict, device=device )

print(species)
print(prob)


