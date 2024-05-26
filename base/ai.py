from torch import nn
import torch
import numpy as np
import pickle
import os

device = "cpu"


class isMushroomClassificationModel(nn.Module):
    """
    Model architecture to classify if there is a mushroom
    in the photo
    """
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*28*28,
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x


# class RestPercepton_block(nn.Module):
#     """
#     Base class for percepton block with residual connection (no pre-activation and BN before conv)

#     in_channels - num. channles into block
#     out_channels - num. concatenated channles out from block
#     conv_size_in - list of num. of channels into 3 and 5 conv (respectively) [conv 1x1 in is always size of in_channels]
#     conv_size_out - list of num. of channels going out from 1,3,5 conv (-||-)
#     stride, padding - list with stride and padding values for 1, 3, 5 conv respectively
#     change_depth_pool - change depth for pooling. By default "False"(no change), if used must be int (out depth from pool section)
#     """


#     #TODO add batch normalization and activation functions after conv
#     def __init__(self, in_channels, out_channels,
#                 conv_size_in:list, conv_size_out:list,
#                 stride:list=[1,1,1], padding:list=[0, 1, 2],
#                 change_depth_pool=False):

#         # checking if dim are correct
#         if(change_depth_pool):
#             if(out_channels != sum(conv_size_out) + change_depth_pool):
#                 raise ValueError(
#                     "Sum of out channels of the block must be equal to sum of out channels of convs inside the block"
#                 )
#         elif(not change_depth_pool):
#             if(out_channels != sum(conv_size_out) + in_channels):
#                 raise ValueError(
#                     "Sum of out channels of the block must be equal to sum of out channels of convs inside the block"
#                 )

#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels


#         # conv 1x1
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels, conv_size_out[0], kernel_size=1, stride=stride[0], padding=padding[0]),
#             nn.BatchNorm2d(conv_size_out[0]),
#             nn.ReLU()
#         )

#         # conv 3x3
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(in_channels, conv_size_in[0], kernel_size=1, padding=0, stride=1), # change depth so it matches 3x3 conv in size
#             nn.Conv2d(conv_size_in[0], conv_size_out[1], kernel_size=3, stride=stride[1], padding=padding[1]),
#             nn.BatchNorm2d(conv_size_out[1]),
#             nn.ReLU()
#         )

#        # conv 5x5
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(in_channels, conv_size_in[1], kernel_size=1, padding=0, stride=1),
#             nn.Conv2d(conv_size_in[1], conv_size_out[2], kernel_size=5, stride=stride[2], padding=padding[2]),
#             nn.BatchNorm2d(conv_size_out[2]),
#             nn.ReLU()
#         )

#         # max pool 3x3
#         if(change_depth_pool):
#             self.pool = nn.Sequential(
#                 nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
#                 nn.Conv2d(in_channels, change_depth_pool, kernel_size=1, padding=0, stride=1),
#                 nn.BatchNorm2d(change_depth_pool),
#                 nn.ReLU()
#             )

#         else:
#             self.pool = nn.Sequential(
#                 nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
#                 nn.BatchNorm2d(self.in_channels),
#                 nn.ReLU()
#             )

#         # changer depth of rest connection
#         if(in_channels != out_channels):
#             self.RestConv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1)

#     def forward(self, x):

#         conv1 = self.conv1(x)
#         conv3 = self.conv3(x)
#         conv5 = self.conv5(x)
#         pool = self.pool(x)

#         if(self.in_channels != self.out_channels):

#             residual = self.RestConv(x)
#         else:
#             residual = x


#         return(torch.cat([conv1, conv3, conv5, pool], dim=1) + residual)

class ImageClassifierNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def train_step(self, input, labels):
        input = input.to(device=device)
        labels = labels.to(device=device)

        preds = self.forward(input)
        labels = labels.unsqueeze(1).float() # pred are 2-dim, so we either have to squeze preds dim1 or unsqueeze labels on dim1
        loss = self.loss(preds, labels)

        return loss

    def val_step(self, input, labels):

        input = input.to(device=device)
        labels = labels.to(device=device)

        preds = self.forward(input)
        labels = labels.unsqueeze(1).float()
        loss = self.loss(preds, labels)
        accuracy = self._accuracy(preds, labels)

        return {"loss":loss.detach(), "accuracy":accuracy}


    def val_epoch_end(self, preformance_measurement_data):

        accuracy = [x["accuracy"] for x in preformance_measurement_data]
        avg_accuracy = np.mean(accuracy)

        loss = [x["loss"].cpu().numpy() for x in preformance_measurement_data]
        avg_loss = np.mean(loss)

        return avg_loss, avg_accuracy


    def _accuracy(self, preds, labels):

        probabilities = torch.sigmoid(preds)

        predicted_labels = (probabilities > .5).float()

        correct = (predicted_labels == labels).float()
        accuracy = correct.sum() / len(preds)

        return accuracy.item()

class RestGoogleNet_Clasificator_biniary(ImageClassifierNetwork):
    """
    Input - in_channels x 96x96
    """
    def __init__(self, in_channels):
        super().__init__()


        self.Intercepton1 = RestPercepton_block(in_channels=32, out_channels=144, conv_size_in=[32,32], conv_size_out=[16, 48, 48],change_depth_pool=False)
        self.Intercepton2 = RestPercepton_block(in_channels=144, out_channels=208, conv_size_in=[64,64], conv_size_out=[24, 72, 64],change_depth_pool= 48)
        self.Intercepton3 = RestPercepton_block(in_channels=208, out_channels=232, conv_size_in=[96,96], conv_size_out=[32, 96, 72],change_depth_pool=32)
        self.Intercepton4 = RestPercepton_block(in_channels=232, out_channels=288, conv_size_in=[128,128], conv_size_out=[48, 128, 96],change_depth_pool=16)

        self.Conv1 = nn.Conv2d(in_channels, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.Conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)

        self.MaxPool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.AvgPool_6x6 = nn.AvgPool2d(kernel_size=8, stride=1)

        self.Dropout = nn.Dropout(0.5)
        self.Linear = nn.Linear(288, 1)

        self.BN1 = nn.BatchNorm2d(16)
        self.BN2 = nn.BatchNorm2d(32)
        self.BN3 = nn.BatchNorm2d(264)

        self.activation = nn.ReLU()


    def forward(self, x):

        x = self.Conv1(x)
        x = self.BN1(x)
        x = self.activation(x)
        x = self.MaxPool_2x2(x) #64x64

        x = self.Conv2(x)
        x = self.BN2(x)
        x = self.activation(x)
        x = self.MaxPool_2x2(x) #32x32

        x = self.Intercepton1(x)
        x = self.MaxPool_2x2(x) # 16x16

        x = self.Intercepton2(x)
        x = self.Intercepton3(x)
        x = self.MaxPool_2x2(x) # 232x8x8

        x = self.Intercepton4(x)
        x = self.AvgPool_6x6(x) # 288x1x1

        x = x.view(x.size(0), -1)

        x = self.Dropout(x)
        x = self.Linear(x)

        return x




class RestGoogleNet_Clasificator_species(ImageClassifierNetwork):
    """
    Input - in_channels x 96x96
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.Conv1 = nn.Conv2d(in_channels, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)

        self.Intercepton1 = RestPercepton_block(in_channels=64, out_channels=168, conv_size_in=[32,32], conv_size_out=[24, 64, 64],change_depth_pool=16)
        self.Intercepton2 = RestPercepton_block(in_channels=168, out_channels=318, conv_size_in=[72,72], conv_size_out=[60, 118, 108],change_depth_pool= 32)
        self.Intercepton3 = RestPercepton_block(in_channels=318, out_channels=414, conv_size_in=[128,96], conv_size_out=[78, 154, 134],change_depth_pool=48)
        self.Intercepton4 = RestPercepton_block(in_channels=414, out_channels=540, conv_size_in=[140,124], conv_size_out=[112, 190, 174],change_depth_pool=64)
        self.Intercepton5 = RestPercepton_block(in_channels=540, out_channels=712, conv_size_in=[192,150], conv_size_out=[152, 256, 224],change_depth_pool=80)

        self.MaxPool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.AvgPool_4x4 = nn.AvgPool2d(kernel_size=4, stride=1)

        self.Dropout = nn.Dropout(0.5)
        self.Linear = nn.Linear(712, num_classes)


        self.BN1 = nn.BatchNorm2d(32)
        self.BN2 = nn.BatchNorm2d(64)

        self.activation = nn.ReLU()


    def forward(self, x):

        x = self.Conv1(x)
        x = self.BN1(x)
        x = self.activation(x)
        x = self.MaxPool_2x2(x) # 128x128

        x = self.Conv2(x)
        x = self.BN2(x)
        x = self.activation(x)
        x = self.MaxPool_2x2(x) # 64x64

        x = self.Intercepton1(x)
        x = self.MaxPool_2x2(x) # 32x32

        x = self.Intercepton2(x)
        x = self.MaxPool_2x2(x) # 16x16
        x = self.Intercepton3(x)
        x = self.MaxPool_2x2(x) # 8x8

        x = self.Intercepton4(x)
        x = self.MaxPool_2x2(x) # 4x4
        x = self.Intercepton5(x)
        x = self.AvgPool_4x4(x)

        x = x.view(x.size(0), -1)

        x = self.Dropout(x)
        x = self.Linear(x)

        return x



# new models to classify mushrooms species 26.05
current_path = os.getcwd()
path = os.path.join(current_path, 'base', 'ai_models', 'classes_member_cnt.pkl')
with open(path, 'rb') as file:
    classes_elem_cnt = pickle.load(file)

max_count = max(classes_elem_cnt.values())

# Oblicz wagi dla każdej klasy
weights = torch.tensor([max_count / classes_elem_cnt[i] for i in range(len(classes_elem_cnt))])




class RestPercepton_block(nn.Module):
    """
    Base class for percepton block with residual connection (no pre-activation and BN before conv)

    in_channels - num. channles into block
    out_channels - num. concatenated channles out from block
    conv_size_in - list of num. of channels into 3 and 5 conv (respectively) [conv 1x1 in is always size of in_channels]
    conv_size_out - list of num. of channels going out from 1,3,5 conv (-||-)
    stride, padding - list with stride and padding values for 1, 3, 5 conv respectively
    change_depth_pool - change depth for pooling. By default "False"(no change), if used must be int (out depth from pool section)
    """


    #TODO add batch normalization and activation functions after conv
    def __init__(self, in_channels, out_channels,
                conv_size_in:list, conv_size_out:list,
                stride:list=[1,1,1], padding:list=[0, 1, 2],
                change_depth_pool=False):

        # checking if dim are correct
        if(change_depth_pool):
            if(out_channels != sum(conv_size_out) + change_depth_pool):
                raise ValueError(
                    "Sum of out channels of the block must be equal to sum of out channels of convs inside the block"
                )
        elif(not change_depth_pool):
            if(out_channels != sum(conv_size_out) + in_channels):
                raise ValueError(
                    "Sum of out channels of the block must be equal to sum of out channels of convs inside the block"
                )

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels


        # conv 1x1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, conv_size_out[0], kernel_size=1, stride=stride[0], padding=padding[0]),
            nn.BatchNorm2d(conv_size_out[0]),
            nn.ReLU()
        )

        # conv 3x3
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, conv_size_in[0], kernel_size=1, padding=0, stride=1), # change depth so it matches 3x3 conv in size
            nn.Conv2d(conv_size_in[0], conv_size_out[1], kernel_size=3, stride=stride[1], padding=padding[1]),
            nn.BatchNorm2d(conv_size_out[1]),
            nn.ReLU()
        )

       # conv 5x5
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, conv_size_in[1], kernel_size=1, padding=0, stride=1),
            nn.Conv2d(conv_size_in[1], conv_size_out[2], kernel_size=5, stride=stride[2], padding=padding[2]),
            nn.BatchNorm2d(conv_size_out[2]),
            nn.ReLU()
        )

        # max pool 3x3
        if(change_depth_pool):
            self.pool = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels, change_depth_pool, kernel_size=1, padding=0, stride=1),
                nn.BatchNorm2d(change_depth_pool),
                nn.ReLU()
            )

        else:
            self.pool = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU()
            )

        # changer depth of rest connection
        if(in_channels != out_channels):
            self.RestConv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1)

    def forward(self, x):

        conv1 = self.conv1(x)
        conv3 = self.conv3(x)
        conv5 = self.conv5(x)
        pool = self.pool(x)

        if(self.in_channels != self.out_channels):

            residual = self.RestConv(x)
        else:
            residual = x


        return(torch.cat([conv1, conv3, conv5, pool], dim=1) + residual)

#######################################33


class ImageClassifierNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=weights)

    def train_step(self, input, labels):
        input = input.to(device=device)
        labels = labels.to(device=device)

        preds = self.forward(input)
        loss = self.loss(preds, labels)

        return loss

    def val_step(self, input, labels):

        input = input.to(device=device)
        labels = labels.to(device=device)

        preds = self.forward(input)
        loss = self.loss(preds, labels)
        accuracy = self._accuracy(preds, labels)

        return {"loss":loss.detach(), "accuracy":accuracy}


    def val_epoch_end(self, preformance_measurement_data):

        accuracy = [x["accuracy"].cpu().numpy() for x in preformance_measurement_data]
        avg_accuracy = np.mean(accuracy)

        loss = [x["loss"].cpu().numpy() for x in preformance_measurement_data]
        avg_loss = np.mean(loss)

        return avg_loss, avg_accuracy


    def _accuracy(self, preds, labels):

        batch_size = len(preds)

        pred_indices = torch.argmax(preds, dim=1)
        return torch.tensor(torch.sum(pred_indices == labels).item() / batch_size)




########################################


class RestGoogleNet_Clasificator(ImageClassifierNetwork):
    """
    Input - in_channels x 96x96
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.Conv1 = nn.Conv2d(in_channels, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)

        self.Intercepton1 = RestPercepton_block(in_channels=64, out_channels=168, conv_size_in=[32,32], conv_size_out=[24, 64, 64],change_depth_pool=16)
        self.Intercepton2 = RestPercepton_block(in_channels=168, out_channels=318, conv_size_in=[72,72], conv_size_out=[60, 118, 108],change_depth_pool= 32)
        self.Intercepton3 = RestPercepton_block(in_channels=318, out_channels=414, conv_size_in=[128,96], conv_size_out=[78, 154, 134],change_depth_pool=48)
        self.Intercepton4 = RestPercepton_block(in_channels=414, out_channels=540, conv_size_in=[140,124], conv_size_out=[112, 190, 174],change_depth_pool=64)
        self.Intercepton5 = RestPercepton_block(in_channels=540, out_channels=712, conv_size_in=[192,150], conv_size_out=[152, 256, 224],change_depth_pool=80)

        self.MaxPool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.AvgPool_4x4 = nn.AvgPool2d(kernel_size=4, stride=1)

        self.Dropout = nn.Dropout(0.5)
        self.Linear = nn.Linear(712, num_classes)


        self.BN1 = nn.BatchNorm2d(32)
        self.BN2 = nn.BatchNorm2d(64)

        self.activation = nn.ReLU()


    def forward(self, x):

        x = self.Conv1(x)
        x = self.BN1(x)
        x = self.activation(x)
        x = self.MaxPool_2x2(x) # 128x128

        x = self.Conv2(x)
        x = self.BN2(x)
        x = self.activation(x)
        x = self.MaxPool_2x2(x) # 64x64

        x = self.Intercepton1(x)
        x = self.MaxPool_2x2(x) # 32x32

        x = self.Intercepton2(x)
        x = self.MaxPool_2x2(x) # 16x16
        x = self.Intercepton3(x)
        x = self.MaxPool_2x2(x) # 8x8

        x = self.Intercepton4(x)
        x = self.MaxPool_2x2(x) # 4x4
        x = self.Intercepton5(x)
        x = self.AvgPool_4x4(x)

        x = x.view(x.size(0), -1)

        x = self.Dropout(x)
        x = self.Linear(x)

        return x
