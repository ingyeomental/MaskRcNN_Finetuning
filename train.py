from utils.engine import train_one_epoch, evaluate
import utils.utils as utils
import torch
from dataset import PennFudanDataset
from data_augmentaion import get_transform
import utils.utils as utils


def main():
    # 이가 없으면 임몸으로
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 2개의 클래스만 가짐(배경과 사람)
    num_classes = 2
    # 데이터셋과 정의된 변환들을 사용
    dataset = PennFudanDataset('PeennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

    # 