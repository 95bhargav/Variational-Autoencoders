import math as m
import random
import torch
import torch.utils.data
import torchvision
import pickle as pkl
import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt


global _COLORS, _DEVICE
_COLORS=["royalblue", "maroon", "green", "black", "orangered", "gray", "teal", "purple", "goldenrod", "mediumslateblue"]
_DEVICE="cuda" if torch.cuda.is_available() else "cpu"

plt.rc('font',family='Times New Roman')
plt.rcParams.update({'figure.max_open_warning': 0})
SMALL_SIZE=13
SMALL_MEDIUM=14
MEDIUM_SIZE=16
BIG_MEDIUM=18
BIG_SIZE=20
plt.rc('font', size=SMALL_MEDIUM)          # controls default text sizes
plt.rc('axes', titlesize=BIG_MEDIUM)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_MEDIUM)    # legend fontsize
plt.rc('figure', titlesize=BIG_SIZE)  # fontsize of the figure title

##########################
##########################
##########################

def GetDatasetPath(folder_name = "Datasets", max_depth = 10):
    """


    Parameters
    ----------
    folder_name : str, optional
        Folder name to look for, first occurence is returned. The default is "Datasets".
    max_depth : int, optional
        Maximum number of parental directories to search. The default is 10.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    Depth = 0
    CurrentDir = os.getcwd()

    while Depth < max_depth:
        if folder_name in os.listdir(CurrentDir):
            break
        CurrentDir = os.path.dirname(CurrentDir)
        Depth += 1
    return os.path.join(CurrentDir, folder_name)

################################################
################################################
################################################

################################################
################################################  
                    
def LoadCifar10(path="Datasets", transforms_train = False, transforms_test = False,\
                minibatch=32, worker=0, normalization = "mean", image_size =32):
    """
    Return training- and testing-dataloaders for the CIFAR10 data set.

    Parameters
    ----------
    path : str, optional
        Accepted values are: 'FolderName', 'code' or a direct file path.\n
        \t 'FolderName':\t will automatically search for the given folder name.\n
        \t 'code':\t\t will automatically store the data set at the code location.\n
        \t str with '/' or '\\': will use the specified location to store the dataset.\n
        The default is "Datasets".
    transforms_train : torchvision.transforms.transforms.Compose, optional
        Complete transformation-composition for training data set. Will use RandomRotation, RandomHorizontalFlop and Normalize if False.
    transforms_test : torchvision.transforms.transforms.Compose, optional
        Complete transformation-composition for testing data set. Will use Normalize if False.
    minibatch : int, optional
        Number of Images per Minibatch. The default is 32.
    worker : int, optional
        Number of worker processes. The default is 0.
    normalization : str, optional
        Which normalization function to use.\n
        \t 'mean':\t\t standardize to zero mean and unit std.\n
        \t '-11':\t\t normalize to the range -1...1.\n
        \t otherwise:\t normalize to the range 0...1.\n
        The default is "mean".
    image_size : int, optional 
        Image size to transform the imge to. if not specified image size is 64.\n


    Returns
    -------
    train_loader :
        DataLoader for training data set.
    test_loader :
        DataLoader for testing data set.

    """
    if normalization == "mean":
        Normalize = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    elif normalization == "-11":
        Normalize = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    else:
        Normalize = torchvision.transforms.Normalize((0, 0, 0), (1, 1, 1))
    
    # setting up default tranformers
    DefaultTransformsTrain = torchvision.transforms.Compose([
                                            torchvision.transforms.Resize(image_size),
                                            torchvision.transforms.RandomRotation(5),
                                            torchvision.transforms.RandomHorizontalFlip(),
                                            torchvision.transforms.ToTensor(),
                                            Normalize])
    DefaultTransformsTest = torchvision.transforms.Compose([
                                            torchvision.transforms.Resize(image_size),  
                                            torchvision.transforms.ToTensor(),
                                            Normalize])
    # setting up folder location
    if type(path) == type("123"):
        if path == "code":
            path = os.path.join(".", "Datasets", "Cifar10")
        elif "/" in path or "\\" in path:
            path = os.path.join(path, "Cifar10")
        else:
            path = os.path.join(GetDatasetPath(path), "Cifar10")
        if not os.path.isdir(path):
            os.makedirs(path)
    else:
        sys.exit("Expected type of path to be str, received %s."%type(path))


    # use given transformers or default ones
    TransformsTrain = transforms_train if transforms_train else DefaultTransformsTrain
    TransformsTest = transforms_train if transforms_train else DefaultTransformsTest

    # load data sets and create data loaders
    train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=TransformsTrain),
            batch_size=minibatch, shuffle=True, pin_memory=True, num_workers=worker)

    test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=TransformsTest),
            batch_size=minibatch, shuffle=False, pin_memory=True, num_workers=worker)
    return train_loader, test_loader

################################################
################################################
################################################


def LoadSTL10(path="Datasets", transforms_train = False, transforms_test = False,\
              minibatch=32, worker=0, normalization = "mean", \
                  train_split= "train+unlabeled", image_size = 64):
    """
    Return training- and testing-dataloaders for the STL10 data set.

    Parameters
    ----------
    path : str, optional
        Accepted values are: 'FolderName', 'code' or a direct file path.\n
        \t 'FolderName':\t will automatically search for the given folder name.\n
        \t 'code':\t\t will automatically store the data set at the code location.\n
        \t str with '/' or '\\': will use the specified location to store the dataset.\n
        The default is "Datasets".
    transforms_train : torchvision.transforms.transforms.Compose, optional
        Complete transformation-composition for training data set. Will use RandomCrop, RandomHorizontalFlop and Normalize if False.
    transforms_test : torchvision.transforms.transforms.Compose, optional
        Complete transformation-composition for testing data set. Will use Normalize if False.
    minibatch : int, optional
        Number of Images per Minibatch. The default is 32.
    worker : int, optional
        Number of worker processes. The default is 0.
    normalization : str, optional
        Which normalization function to use. \n
        \t 'mean':\t\t standardize to zero mean and unit std.\n
        \t '-11':\t\t normalize to the range -1...1.\n
        \t otherwise:\t normalize to the range 0...1.\n
        The default is "mean". \n
    train_split : str, optional 
        Splits for training Data.\n
        \t ‘train’, \t ‘unlabeled’, \t ‘test’, \t ‘train+unlabeled’ \n
    image_size : int, optional 
        Image size to transform the imge to. if not specified image size is 64.\n

    Returns
    -------
    train_loader :
        DataLoader for training data set.
    test_loader :
        DataLoader for testing data set.

    """
    if normalization == "mean":
        Normalize = torchvision.transforms.Normalize((0.4384, 0.4314, 0.3989), (0.2647, 0.2609, 0.2741))
    elif normalization == "-11":
        Normalize = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    else:
        Normalize = torchvision.transforms.Normalize((0, 0, 0), (1, 1, 1))
    
    # setting up default tranformers
    DefaultTransformsTrain = torchvision.transforms.Compose([
                                            torchvision.transforms.Resize(image_size),
                                            torchvision.transforms.RandomHorizontalFlip(),
                                            torchvision.transforms.ToTensor(),
                                            Normalize])
    
    DefaultTransformsTest = torchvision.transforms.Compose([
                                            torchvision.transforms.Resize(image_size),
                                            torchvision.transforms.ToTensor(),
                                            Normalize])
    # setting up folder location
    if type(path) == type("123"):
        if path == "code":
            path = os.path.join(".", "Datasets", "STL10")
        elif "/" in path or "\\" in path:
            path = os.path.join(path, "STL10")
        else:
            path = os.path.join(GetDatasetPath(path), "STL10")
        if not os.path.isdir(path):
            os.makedirs(path)
    else:
        sys.exit("Expected type of path to be str, received %s."%type(path))


    # use given transformers or default ones
    TransformsTrain = transforms_train if transforms_train else DefaultTransformsTrain
    TransformsTest = transforms_train if transforms_train else DefaultTransformsTest

    # load data sets and create data loaders
    train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.STL10(root=path, split= train_split, download=True, transform=TransformsTrain),
            batch_size=minibatch, shuffle=True, pin_memory=True, num_workers=worker)

    test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.STL10(root=path, split="test", download=True, transform=TransformsTest),
            batch_size=minibatch, shuffle=False, pin_memory=True, num_workers=worker)
    return train_loader, test_loader

################################################
################################################
################################################
 
def MovingAverage(data, window=-1):
    cumsum = np.cumsum(np.insert(data, 0, 0)) 
    return (cumsum[window:] - cumsum[:-window]) / float(window)

################################################
################################################
################################################
    
class DynReport():
    def __init__(self, names, precision, average=10, show=25, line=250, header=5000, plotsize=4, autoshow=True, step=1):
        self.NAMES=names[:]
        self.AVERAGE=average
        self.SHOW=show
        self.LINE=line
        self.HEADER=header
        self.AUTOSHOW=autoshow
        self.BATCH=0
        self.STEP=step
        
        if "Loss" in self.NAMES:
            self.NAMES.append("avg. Loss")
        if "Acc" in self.NAMES:
            self.NAMES.append("avg. Acc")
        self.LENS=[len(self.NAMES[idx])+2+precision for idx in range(len(self.NAMES))]
        self.VALS={}
        for name in self.NAMES:
            self.VALS[name]=[]
        self.COLS=len(self.NAMES)
        self.PRECISION=precision  
        self.PLOTSIZE=plotsize
        self.DUMPDICT={}
        
        
    def SetValues(self, vals):
        if self.BATCH%self.STEP==0:
            for col in range(self.COLS):
                name=self.NAMES[col]
                if name=="avg. Loss":
                    self.VALS[name].append(torch.sum(torch.tensor(self.VALS["Loss"][-self.AVERAGE:]))/self.AVERAGE)
                elif name=="avg. Acc":
                    self.VALS[name].append(torch.sum(torch.tensor(self.VALS["Acc"][-self.AVERAGE:]))/self.AVERAGE)
                else:
                    self.VALS[name].append(vals[col])
                    
            if self.AUTOSHOW:
                if self.BATCH==0:
                    self.GetHead()
                    self.Show()
                    self.NewLine()
                else:
                    if self.BATCH%self.SHOW==0:
                        self.Show()
                    if self.BATCH%self.LINE==0:
                        self.NewLine()
                    if self.BATCH%self.HEADER==0:
                        self.GetHead()
        self.BATCH+=1          
    
    def Show(self):
        outstr="\r"
        args=[]
        for col in range(self.COLS):
            val=self.VALS[self.NAMES[col]][-1]  
            outstr+="{:s}"
            
            if type(val)==float:
                val=str(round(val, self.PRECISION))
            elif type(val)==torch.Tensor:
                val=str(round(val.item(), self.PRECISION))
            else:
                val=str(val)
            args.append(val+(self.LENS[col]-len(val))*" ")
        print(outstr.format(*args), end="")
        
    def GetHead(self):
        self.NewLine()
        string=""
        for col in range(self.COLS):
            name=self.NAMES[col]
            string+=name+(self.LENS[col]-len(name))*" "
        print(string)
        
    def NewLine(self):
        print("")      



class FigureDict():
    def __init__(self, folder, dpi=200, saving=False):
        super(FigureDict, self).__init__()
        self.FIGS=[]
        self.NAMES=[]
        self.DPIS=[]
        self.FOLDER=folder
        self.DPI=dpi
        self.SAVING=saving
    
    def StoreFig(self, fig, name=False, dpi=False, saving=False):
        if str(type(fig))=="<class 'matplotlib.figure.Figure'>":
            self.FIGS.append(fig)
            self.DPIS.append(dpi if dpi else self.DPI)
            if type(name)==str:
                self.NAMES.append(name)
            else:
                self.NAMES.append("NoName")
            
            if saving or self.SAVING:
                self.SaveOneFig(fig=fig, name=name, dpi=self.DPIS[-1], prints=False)
        else:
            print("No Figure given, skipping.")
            
    def SaveAllFigs(self, dpi=False, prints=False):
        for fignum in range(len(self.FIGS)):
            dpi=self.DPIS[fignum]
            fig=self.FIGS[fignum]
            name=self.NAMES[fignum]
            if fig and name:
                if name[-4:]==".png":
                    fullpath=os.path.join(self.FOLDER, name)
                else:
                    fullpath=os.path.join(self.FOLDER, name+".png")
                
                foldername=os.path.dirname(fullpath)
                if not os.path.exists(foldername):
                    print("creating %s"%foldername)
                    os.makedirs(foldername)
                fig.savefig(os.path.join(self.FOLDER, name), dpi=dpi)
                if prints:
                    print("%s saved."%(name))
        
    def SaveOneFig(self, fig, name, dpi=False, prints=False):
        if fig and name:
            if name[-4:]==".png" or name[-4:]=="svg":
                fullpath=os.path.join(self.FOLDER, name)
            else:
                fullpath=os.path.join(self.FOLDER, name+".png")
            foldername=os.path.dirname(fullpath)
            if not os.path.exists(foldername):
                print("creating %s"%foldername)
                os.makedirs(foldername)
            fig.savefig(os.path.join(self.FOLDER, name),bbox_inches = "tight", dpi=dpi)
            if prints:
                print("%s saved."%(name))
        

################################################
################################################
################################################














    
    