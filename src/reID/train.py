import numpy as np

import torch

from dataset import PlushieTrainDataset
from model import SiameseNetwork
from transforms import Transforms
from utils import DeviceDataLoader, accuracy, get_default_device, to_device

def loss_batch(model, loss_func, anchor, image, label, opt=None, metric=None): # Update model weights and return metrics given xb, yb, model
    preds = model(anchor, image) 
    loss = loss_func(preds, label.unsqueeze(1).float()) 
    
    if opt is not None: # if no optimiser
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    metric_result = None
    if metric is not None: # if no metric
        metric_result = metric(preds, label)
        
    return loss.item(), len(anchor), metric_result # loss value, number of samples, metric result


def fit(epochs, model, loss_func, train_dl, val_dl, opt_func=torch.optim.SGD, lr=0.01, metric=None): #training function, SGD = Stochastic Gradient Descent (being used to train)
    train_losses, val_losses, val_metrics = [] , [], [] #empty lists for storing
    
    opt = opt_func(model.parameters(), lr=lr) # lr = learning rate
    
    for epoch in range(1, epochs+1): # iterating over each epoch
        model.train() # Setting for pytorch - training mode
        for anchor,image,label in train_dl:
            train_loss, _, _ = loss_batch(model, loss_func, anchor, image, label, opt) # update weights
            
        model.eval() # Setting - eval mode, calls evaluate function to calculate
        val_loss, total, val_metric = evaluate(model, loss_func, val_dl, metric) 
        
        # appends respective data to respective lists
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)
        
        #prints epoch number and data
        if metric is None:
            print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(
            epoch, train_loss, val_loss))
        else:
            print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_{}: {:.4f}".format(
            epoch, train_loss, val_loss, metric.__name__, val_metric))
            
    return train_losses, val_losses, val_metrics


def evaluate(model, loss_func, val_dl, metric=None): # evaluates the model
    with torch.no_grad(): #disable gradient calculation
        results = [loss_batch(model, loss_func, anchor, image, label, metric=metric) for anchor, image, label in val_dl] # calls loss_batch to calculate loss and metric
        
        losses, nums, metrics = zip(*results)
        # sample count
        total = np.sum(nums)
        
        #calculates the average
        avg_loss = np.sum(np.multiply(losses, nums)) / total

        avg_metric = None
        if metric is not None:
            avg_metric = np.sum(np.multiply(metrics, nums)) / total
            
        return avg_loss, total, avg_metric





def main(): # starting point of program
    # variable declaration
    train_filepath = ""
    train_img_dir = ""
    val_filepath = ""
    val_img_dir = ""
    train_bs = 32
    test_bs = 5
    num_epochs = 5
    lr = 0.005
    val_ratio = 0.2
    
    # enables anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    # data preperation
    transform = Transforms()
    train_dataset = PlushieTrainDataset(filepath=train_filepath, img_dir=train_img_dir, transform=transform)
    valid_dataset = PlushieTrainDataset(filepath=val_filepath, img_dir=val_img_dir, transform=transform)
    network = SiameseNetwork()
    

    print("The length of Train set is {}".format(len(train_dataset)))
    print("The length of Valid set is {}".format(len(valid_dataset)))

    # data loader
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=4)
    val_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=test_bs, shuffle=True, num_workers=4)

    # device placement
    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    to_device(network, device)
  
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam

    # model training
    train_losses, val_losses, val_metrics = fit(num_epochs, network, criterion, 
                                            train_dl, val_dl, optimizer, lr, accuracy)

    # model saving
    torch.save(network.state_dict(), 'model.pth')

    
if __name__ == "__main__":
    main()
