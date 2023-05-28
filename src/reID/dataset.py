import os
import matplotlib.pyplot as plt

from PIL import Image
import cv2

from torch.utils.data import Dataset #Pytorch class for handling dataset
from transforms import Transforms #Importing from transforms.py




class PlushieTrainDataset(Dataset):
    
    #Read the file, extract sample, stores them in self.samples
    def __init__(self, filepath, img_dir, transform=None): #Store within class, initialise image data set, directory containing image, transform (if any)
        self.samples = []
        self.img_dir = img_dir
        self.transform = transform

        with open(filepath, 'r') as f:
            self.samples = [line.strip() for line in f]


    def __len__(self):  #Returns the length, the no. of samples
        return len(self.samples)

    def __getitem__(self, i): #Retrieves an item from the dataset at i
        line = self.samples[i].split()
        if len(line) == 3: #Getting the amount of data per line
            anchor_name, anchor_num, img_num = line
            img_name = anchor_name
            is_same = 1
        elif len(line) == 4:
            anchor_name, anchor_num, img_name, img_num = line
            is_same = 0
        else:
            print(len(line), line)
            raise Exception("Shouldn't be here")
        
        anchor = cv2.imread(os.path.join(self.img_dir, str(anchor_name), f"{anchor_name}_{anchor_num}.png")) #Loads anchor image
        img = cv2.imread(os.path.join(self.img_dir, img_name, f"{img_name}_{img_num}.png")) #Loads another image (Anchor image if len(line) = 3)
        
        if self.transform: #If there is transformation
            anchor = self.transform(anchor)
            img = self.transform(img)

        return anchor, img, is_same



def main(): #Usage of dataset
    t = Transforms()
    filepath = ""
    img_dir = ""
    d = PlushieTrainDataset(filepath=filepath, img_dir=img_dir, transform=t)
    
    e = d[0] #Retrieve first item (line 48)
    axs = plt.figure(figsize=(9, 9)).subplots(1, 2) #Creates 9x9 figure
    plt.title(e[2]) #is_same
    axs[0].imshow(e[0].permute(1,2,0)) #anchor, show image, adjust dimentions
    axs[1].imshow(e[1].permute(1,2,0)) #img, show image, adjust dimensions

if __name__ == "__main__": #If script is main module
    main()
