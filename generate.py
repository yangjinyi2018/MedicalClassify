import numpy as np
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import scipy.io as io
from matplotlib import pyplot as plt


class mydata(Dataset):
    def __init__(self, x_train_path, y_train_path=None, transform=None):
        # Transforms
        self.file = os.listdir(x_train_path)
        self.file.sort(key=lambda x: int(x[9:-4]))
        self.x_path = x_train_path
        if (y_train_path):
            self.y_path = y_train_path
        if (transform):
            self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 读入x_train
        image_path = os.path.join(self.x_path, self.file[idx])

        tmp = np.load(image_path)

        myarray1 = np.array(tmp['seg'].astype(float))
        myarray2 = np.array(tmp['voxel'].astype(float))/255
        #分割图像和原图像取交集
        myarray=myarray1*myarray2
        #中间部分
        myarray = myarray[34:66, 34:66, 34:66]
        img = torch.from_numpy(myarray)


        img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)


        return (img)

    def __len__(self):
        return len(self.file)

x_train_path = './train_val' 
x_test_path = './test'

x_train_path = 'nodule' 
x_test_path = 'test'

traindata = mydata(x_train_path)
testdata = mydata(x_test_path)
train_dataloader = DataLoader(traindata, batch_size=1, shuffle=False)
test_dataloader = DataLoader(testdata, batch_size=1, shuffle=False)


test=np.zeros((117,32,32,32))
for i, data in enumerate(test_dataloader, 0):
    img = data
    img=np.array(img[0][0])
    test[i]=np.array(img)
test=np.array(test)

train=np.zeros((465,32,32,32))
for i, data in enumerate(train_dataloader, 0):
    img = data
    img=np.array(img[0][0])
    train[i]=np.array(img)
train=np.array(train)

io.savemat('generateDone.mat',{'train':train,'test':test})
