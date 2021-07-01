from utils.engine import train_one_epoch
import torch
import torchvision
from dataset import PennFudanDataset
from data_augmentaion import get_transform
import utils.utils as utils


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn
)
