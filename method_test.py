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

# train
images, targets = next(iter(data_loader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]
output = model(images, targets)

# inference
model.eval()
inference_input = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(inference_input)