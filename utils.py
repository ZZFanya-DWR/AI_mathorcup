import numpy as np
import torch
import torch.functional as F



class ToTensor(object):
    def __call__(self, sample):
        img = sample['img']
        mask = sample['label']
        img = np.expand_dims(img, axis = 0).astype(np.float32)
        mask = np.expand_dims(mask, axis = 0).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'img' : img,
                'label' : mask
                }


class GraytoRgb(object):
    def __call__(self, sample):
        img = sample['img']
        mask = sample['label']
        img = img.repeat(3,1,1)
        return {'img' : img,
                'label' : mask
                }

class RandomHorizontalFlip(object): #左右翻转
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


def read_tif(img_dir):
    dataset = gdal.Open(img_dir)
    width = dataset.RasterXSize                         # 获取数据宽度
    height = dataset.RasterYSize                        # 获取数据高度
    outbandsize = dataset.RasterCount                   # 获取数据波段数
    im_geotrans = dataset.GetGeoTransform()             # 获取仿射矩阵信息
    im_proj = dataset.GetProjection()                   # 获取投影信息
    datatype = dataset.GetRasterBand(1).DataType
    im_data = dataset.ReadAsArray()                     #获取数据 
    img3 = uint16to8(im_data)
    img2 = img3[0:3,:,:]
    # out = img2[:,:,::-1] #rgb->bgr
    return img2

#拉伸图像  #图片的16位转8位
def uint16to8(bands, lower_percent=0.001, higher_percent=99.999): 
    out = np.zeros_like(bands,dtype = np.uint8)
    n = bands.shape[0] 
    for i in range(n): 
        a = 0 # np.min(band) 
        b = 255 # np.max(band) 
        c = np.percentile(bands[i, :, :], lower_percent) 
        d = np.percentile(bands[i, :, :], higher_percent) 
        t = a + (bands[i, :, :] - c) * (b - a) / (d - c) 
        t[t<a] = a
        t[t>b] = b
        out[i, :, :] = t 
    return out 
