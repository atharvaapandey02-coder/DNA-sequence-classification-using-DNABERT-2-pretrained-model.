import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms

# Normalization constants
SHEBA_MEAN = 0.1572722182674478
SHEBA_STD = 0.16270082671743363

# Define the custom block for the CNN model
class CBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBlock, self).__init__()
        assert out_channels % 4 == 0
        # Convolution layers with different kernel sizes
        self.conv3 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels // 4, 5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, out_channels // 4, 7, padding=3)
        self.conv9 = nn.Conv2d(in_channels, out_channels // 4, 9, padding=4)
        self.bn = nn.BatchNorm2d(out_channels)  # Batch normalization
        self.act = nn.ReLU()  # Activation function

    def forward(self, x):
        # Concatenate the outputs from different convolution layers
        return self.act(self.bn(torch.cat([self.conv3(x), self.conv5(x), self.conv7(x), self.conv9(x)], 1)))

# Define the CNN model for calcification detection
class NetC(nn.Module):
    def __init__(self):
        super(NetC, self).__init__()
        self.block1 = CBlock(1, 4)
        self.block2 = CBlock(4, 16)
        self.block3 = CBlock(16, 32)
        self.block4 = CBlock(32, 64)
        self.block5 = CBlock(64, 128)
        self.pred = nn.Conv2d(128, 1, 5, padding=2)  # Final prediction layer

    def forward(self, x):
        # Forward pass through all blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return self.pred(x)

# Load and preprocess mammogram image
def load_mamm(case_path):
    mamm = cv2.imread(case_path).astype(np.float32) / 255  # Read and normalize image
    mamm = mamms_preprocess(mamm)  # Apply preprocessing
    return mamm

# Preprocessing of mammogram image
def mamms_preprocess(mamm):
    # Clean and prepare the mammogram image
    mamm = clean_mamm(mamm)
    act_w = get_act_width(mamm)
    mamm = cut_mamm(mamm, act_w)
    return mamm

# Clean the mammogram image
def clean_mamm(mamm):
    # Apply basic cleaning (e.g., zero out borders)
    mamm[:10, :] = 0
    mamm[-10:, :] = 0
    mamm[:, -10:] = 0
    msk1 = (mamm[:, :, 0] == mamm[:, :, 1]) & (mamm[:, :, 1] == mamm[:, :, 2])
    mamm = mamm.mean(axis=2) * msk1  # Convert to grayscale if possible
    msk = np.uint8((mamm > 0) * 255)
    msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50)))
    comps = cv2.connectedComponentsWithStats(msk)
    common_label = np.argmax(comps[2][1:, cv2.CC_STAT_AREA]) + 1
    msk = (comps[1] == common_label).astype(np.uint8)
    mamm *= msk
    return mamm

# Calculate the actual width of the mammogram (region of interest)
def get_act_width(mamm):
    w = mamm.shape[1] // 3
    while mamm[:, w:].max() > 0:
        w += 1
    return w

# Cut the mammogram to focus on the relevant region
def cut_mamm(mamm, act_w):
    h = mamm.shape[0]
    mamm = mamm[:h, :act_w]
    return mamm

# Prediction function using the trained model
def predict(net, img):
    model_device = next(net.parameters()).device  # Get the model's device (CPU/GPU)
    sig = nn.Sigmoid()  # Sigmoid activation for binary classification
    toten = transforms.ToTensor()  # Convert image to tensor
    norm = transforms.Normalize(mean=[SHEBA_MEAN], std=[SHEBA_STD])  # Normalize image

    # Add channel dimension if missing
    if len(img.shape) == 2:
        img = img[..., None]

    img = norm(toten(img).float())[:1][None, ...].float()  # Normalize and convert to tensor
    img = img.to(model_device)  # Move tensor to the model's device

    with torch.no_grad():
        pred = net(img)  # Make prediction

    sig_pred = sig(pred)  # Apply sigmoid to get probabilities
    return sig_pred[0, 0].cpu().numpy()

# Save the mammogram with bounding boxes around predicted calcifications
def save_mamm_w_boxes(processed_mamm, prediction, filename, th=.5):
    # Create a 3-channel image from the processed mammogram
    result = (np.tile(processed_mamm[..., None], (1, 1, 3)) * 255).astype('uint8')
    bbs = np.zeros_like(result)  # Empty image for bounding boxes

    # Find connected components (detected calcifications)
    cc = cv2.connectedComponentsWithStats((prediction > th).astype('uint8'), 8)
    for i in range(1, cc[0]):
        start_point = cc[2][i][0] - 5, cc[2][i][1] - 5
        end_point = start_point[0] + cc[2][i][2] + 10, start_point[1] + cc[2][i][3] + 10
        # Draw bounding box
        cv2.rectangle(bbs, start_point, end_point, (0, 0, 255), 2)

    # Combine the original image and the bounding boxes
    result_with_boxes = cv2.addWeighted(result, 1.0, bbs, 0.5, 0)
    cv2.imwrite(filename, result_with_boxes)