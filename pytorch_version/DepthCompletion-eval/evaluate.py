import os
import torch
from torch.utils.data import DataLoader
import cv2 as cv
from dataloader import NyuV2Dataset
import numpy as np

DATA_DIR = 'Data'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    with torch.no_grad():

        file_list = ['%s/%s' % (DATA_DIR, i) for i in os.listdir(DATA_DIR)]
        c_list = [i for i in file_list if 'col' in i]
        r_list = [i for i in file_list if 'raw' in i]
        fn_list = [i.split('/')[-1].split('-')[0] for i in c_list]
        len_ = len(c_list)

        test_list = ['%s,%s\n' % (c_list[i], r_list[i]) for i in range(len_)]

        test_dataset = NyuV2Dataset(test_list)

        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

        model = torch.load('Checkpoints\\model.pkl').to(device)

        print('Start evaluate...')
        for idx, _input in enumerate(test_loader):

            model.eval()
            image = torch.autograd.Variable(_input.float().to(device))

            output = model(image).cpu().numpy().squeeze()*1000.
            output = output.astype(np.uint16)

            cv.imwrite('Result\\%s-result.png' % fn_list[idx], output)
        print('Finished')


if __name__ == '__main__':
    main()
#