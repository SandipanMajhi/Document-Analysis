import torch
import torch.nn
import torchvision
import torchvision.datasets as datasets
import numpy as np
import cv2

train_arrays = []
train_labels = []
test_arrays = []
test_labels = []

device = "cuda" if torch.cuda.is_available() else "cpu"

def pil_to_np(stock_set):
    target_threshold = []
    target_set = []
    target_labels = []
    for i in range(0, len(stock_set)):
        img = np.array(stock_set[i][0], dtype = np.uint8)
        thresh , im_bw = cv2.threshold(img,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) 
        target_threshold.append(thresh)
        target_set.append(np.array(im_bw, dtype = np.uint8))
        target_labels.append(stock_set[i][1])

    return target_threshold, target_set, target_labels


class Data_Maps():
    def __init__(self, dataset):
        self.dataset = dataset
        
    def get_maps(self, step1, step2):
        l = self.left_maps(step1, step2)
        l_diff = torch.FloatTensor(self.map_diff(l))
        l_diff = l_diff / torch.max(l_diff)
        r = self.right_maps(step1, step2)
        r_diff = torch.FloatTensor(self.map_diff(r))
        r_diff = r_diff / torch.max(r_diff)
        t = self.top_maps(step2, step1)
        t_diff = torch.FloatTensor(self.map_diff(t))
        t_diff = t_diff / torch.max(t_diff)
        b = self.top_maps(step2,step1)
        b_diff = torch.FloatTensor(self.map_diff(b))
        b_diff = b_diff / torch.max(b_diff)
        
        return l, l_diff, r, r_diff, t, t_diff, b, b_diff
      
  
    def map_diff(self, cmaps):
        maps = []
        for i in range(len(cmaps)):
            sz = len(cmaps[i])
            temp = cmaps[i][0]
            for j in range(0,len(cmaps[i])):
                cmaps[i][j] = (cmaps[i][j] - cmaps[i][(j+1)%sz])
            cmaps[i][-1] = (cmaps[i][-1] - temp)
        return cmaps

    def left_maps(self, rstep, cstep):
        sz = len(self.dataset)
        left_maps = []
        for i in range(sz):
            lmap = []
            for row in range(0,self.dataset[i].shape[0],rstep):
                count = 0
                for col in range(0,self.dataset[i].shape[1],cstep):
                    if self.dataset[i][row][col] == 0:
                        count = count + 1
                    else:
                        break
                lmap.append(count)
        left_maps.append(lmap)
        return left_maps

    def right_maps(self, rstep, cstep):
        sz = len(self.dataset)
        right_maps = []
        for i in range(sz):
            rmap = []
            for row in range(0,self.dataset[i].shape[0],rstep):
                count = 0
                for col in range(self.dataset[i].shape[1]-1,-1,-1 * cstep):
                    if self.dataset[i][row][col] == 0:
                        count = count + 1
                    else:
                        break
                rmap.append(count)
        right_maps.append(rmap)
        return right_maps

    def top_maps(self, rstep, cstep):
        sz = len(self.dataset)
        top_maps = []
        for i in range(sz):
            tmap = []
            for col in range(0,self.dataset[i].shape[1],cstep):
                count = 0
                for row in range(0,self.dataset[i].shape[0], rstep):
                    if self.dataset[i][row][col] == 0:
                        count = count + 1
                    else:
                        break
                tmap.append(count)
        top_maps.append(tmap)
        return top_maps
  
    def bottom_maps(self, rstep, cstep):
        sz = len(self.dataset)
        b_maps = []
        for i in range(sz):
            bmap = []
            for col in range(0,self.dataset[i].shape[0],cstep):
                count = 0
                for row in range(self.dataset[i].shape[1]-1,-1,-1 * rstep):
                    if self.dataset[i][row][col] == 0:
                        count = count + 1
                    else:
                        break
                bmap.append(count)
        b_maps.append(bmap)
        return b_maps

   
    
    