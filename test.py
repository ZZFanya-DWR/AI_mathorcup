from modeling.deeplab import DeepLab
from dataset import MyDataset
from torch.utils.data import DataLoader
import torch



if __name__ == "__main__":
    model = DeepLab(backbone='resnet', output_stride=16, num_classes=2)
    model.eval()
    data = MyDataset('data/mathorcup/img', 'data/mathorcup/mask')
    dataload = DataLoader(data, batch_size=1,shuffle=True, num_workers=0)
    for i, data in enumerate(dataload):
        # print(i, data)
        input = data[0]
        output = model(input)
        print(input,output)
    # print(data[0])
    # model.eval()
    # input = torch.rand(1, 3, 513, 513)
    # output = model(input)
    # print(output.size())

