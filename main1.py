# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 21:21:18 2020

@author: Shukla
"""
#%% 1. Importing Libraries
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import time
from Functions import  PlotViz
import nets 
import HelperFunctions as HelpFunc
from shutil import copy as CopyFile 



#%% 2. Parameter Initializations

version         = '-ver_1.190--27-01-2021'
changes         = " Using new idea of not scaling to 1x1 features and 10 label channels"
dataset         = "stl10"
EncOptim        = "Adam"

global _DEVICE
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

feature_length      = 40
num_classes         = 10
Mean                = (0.5,0.5,0.5)
Std                 = (0.5,0.5,0.5)
Batch_size          = 64 
n_epoch             = 1000
Img_size            = 64
Base                = 64
step_E              = 450
gammaE              = 0.1
lrE                 = 0.001
label_recon         = True


#%% 3. Plot settings
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.rc('font',family= 'Times New Roman')
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
plt.rcParams['axes.unicode_minus'] = False
#%% 4. Folder Creation

time_str    = time.strftime("%y_%m_%d__%H_%M_%S"+version)
path        = os.path.dirname(os.getcwd())
folder      = dataset+'_'
 
trainpath   = os.path.join(path,"01_GAN_with_Encoder", folder)
trainfolder = os.path.join(trainpath,time_str)
if not os.path.exists(trainfolder):
    os.makedirs(trainfolder)                        


#%% 5. Training loop and testing loop

def test(ae, fixed_data, criterionMSE, epoch):
    """
    Testing function for the networks.
    
    Inputs:
        encoder         : Encoder network. \n
        Generator       : Generator network. \n
        fixed_data      : Testing DataLoader. \n
        epoch           : epoch number. \n
        
    Returns:
        classifier accuracy, classifier loss,AE loss, labels_list, \
            latent_code, pred_list, class_out
    
    """
    with torch.no_grad():
        
        ae_loss =0
        ae.eval()
        for i,(image,label) in enumerate(fixed_data):
            image = image.to(_DEVICE)
            label = label.to(_DEVICE)
            one_hot_labels = torch.nn.functional.one_hot(label, num_classes).float()
            test_img       = ae.forward(image, one_hot_labels,
                                        mode = "one_hot", 
                                        train_mode= "ae")
            
            mse_loss        = criterionMSE(test_img, image)
            ae_loss += mse_loss.item()
    loss_ae                 =  (ae_loss/len(fixed_data))
    
    return loss_ae
       
def train(CurrentBatch, epoch, ae, optE, criterionMSE, train_data,
          fixed_data, trainfolder,  Reporter):
    """
    Train function to train the networks.
    
    Inputs: 
        CurrentBatch    : Batch number. \n
        epoch           : epoch number. \n
        Encoder         : Encoder network. \n
        Generator       : Generator network. \n
        optE            : Optimizer for Encoder. \n
        optG            : Optrimizer for Generator. \n
        criterionMSE    : Mean-squared Error loss. \n
        train_data      : Training DataLoader. \n
        fixed_data      : Testing DataLoader. \n
        trainfolder     : Location to store the plots. \n
        Reporter        : Reporter object to show the progress. \n
        
    Returns:
        CurrentBatch+1
    """
    
    
    
    for i, (image,label) in enumerate(train_data):
        ae.train()
        image = image.to(_DEVICE)
        label = label.to(_DEVICE)
        
        ## Traditional Autoencoder training loop
        optE.zero_grad()      
        one_hot_labels = torch.nn.functional.one_hot(label, num_classes).float()
        recon_img       = ae(image, one_hot_labels, mode = "one_hot")
        mse_loss        = criterionMSE(recon_img,image)
        mse_loss.backward()
        
        optE.step()
        
        # Testing loop built-in
        if CurrentBatch % 500 == 0:
            loss_ae_test= test(ae, fixed_data, criterionMSE, epoch)
                
        ## Plotting reconstructions
            with torch.no_grad():
                ae.train()
                imag, label            = plot_data["image"], plot_data["label"]
                one_hot_labels  = torch.nn.functional.one_hot(label, num_classes).float()
                recon           = ae.forward(imag, one_hot_labels,
                                             mode = "one_hot",
                                             train_mode = "ae")

                img_title_list  = ["Real","Recon"]
                PlotTitle       = "train_epoch_"+str(epoch)
                FigureDict      = HelpFunc.FigureDict(os.path.join(trainfolder,"train_plots"),dpi =300 )
                PlotViz(imag, recon, img_title_list, FigureDict, 
                        PlotTitle, "cifar10")
            
                ae.eval()
                image, label           = plot_data['imageT'], plot_data['labelT'] 
                one_hot_labels  = torch.nn.functional.one_hot(label, num_classes).float()
                recon_image           = ae.forward(image, one_hot_labels,
                                                   mode = "one_hot",
                                                   train_mode ="ae")

                img_title_list  = ["Real","Recon"]
                PlotTitle       = "test_epoch_"+str(epoch)
                FigureDict      = HelpFunc.FigureDict(os.path.join(trainfolder,"test_plots"),dpi =300) 
                PlotViz(image, recon_image, img_title_list, FigureDict, 
                        PlotTitle, "cifar10")
        else:
            loss_ae_test = loss_test[-1]
        
        loss_test.append(loss_ae_test)
        
        
        ## Reporting   
        Reporter.DUMPDICT['Lr_AE'].append(optE.param_groups[0]['lr'])
        
        Reporter.SetValues([epoch+1, CurrentBatch,
                            (time.time()-Reporter.DUMPDICT["starttime"])/60,
                            mse_loss.item(), loss_test[-1]])
        CurrentBatch+=1
    return CurrentBatch


#%% 6. Loading Dataset
data_path = os.path.dirname(path)
if dataset == "cifar10":
    trainloader,testloader = HelpFunc.LoadCifar10(path =os.path.join(data_path, "Datasets"),
                                              minibatch = Batch_size,
                                              normalization = "-11",
                                              image_size = Img_size)
elif dataset == "stl10":
    trainloader,testloader = HelpFunc.LoadSTL10(path =os.path.join(data_path, "Datasets"),
                                                minibatch = Batch_size,
                                                normalization ="-11",
                                                train_split ='train',
                                                image_size = Img_size)

print('Loading Data.....')

imageT,labelT = next(iter(testloader))
image, label = next(iter(trainloader))
plot_data = dict(
                image  = image.to(_DEVICE),
                label  = label.to(_DEVICE),
                imageT = imageT.to(_DEVICE),
                labelT = labelT.to(_DEVICE),
                 )
loss_test = []
print(f" Dataset : {dataset} \n","len of dataset : ", len(trainloader)*Batch_size, \
      "\n Total batches in one Epoch : ",len(trainloader), "\n")


#%% 7. Object Initializations 

criterionMSE    = nn.MSELoss() 

ae = nets.autoencoder_new(Base,feature_length,image_size =Img_size, label_recon = label_recon).to(_DEVICE)    

optE = torch.optim.Adam(ae.parameters(), lr = lrE)
 

schedE = torch.optim.lr_scheduler.StepLR(optE, step_size=step_E, gamma=gammaE)


print("\n",ae,"\n")
print("\n Changes : ", changes)
print("\n Dataset", dataset)
print("\n Starting Training Process......")   


#%% 9. Training
ReporterNames=['Epoch',"Batch", "Time", "LossMSE", "AE_TestLoss"]

ReporterPrecision=3
ReporterShow=25
ReporterAverage=100
ReporterLine=500
ReporterHeader=5000
ReporterAutoShow=True
ReporterStep=1
Reporter=HelpFunc.DynReport(names=ReporterNames,
                            precision=ReporterPrecision,
                            show=ReporterShow,
                            average=ReporterAverage,
                            line=ReporterLine,
                            header=ReporterHeader,
                            autoshow=ReporterAutoShow,
                            step=ReporterStep)
CurrentBatch                            =   0

Reporter.DUMPDICT["starttime"]      =   time.time()
Reporter.DUMPDICT['Lr_AE']         =   []


epoch = 0
for epoch in range(n_epoch):
    CurrentBatch=train(CurrentBatch, epoch, ae, optE, criterionMSE,
                   trainloader, testloader ,trainfolder, Reporter)
    schedE.step()

    
#%% 10. Plotting 

##############################   
FigureDict = HelpFunc.FigureDict(trainfolder,dpi =300 )


MSE_loss= HelpFunc.MovingAverage(Reporter.VALS['LossMSE'],window=n_epoch)
AE_TestLoss= HelpFunc.MovingAverage(Reporter.VALS['AE_TestLoss'],window=n_epoch)      

FigureLoss1=plt.figure(figsize=(8,5))
plt.plot(MSE_loss, label='MSE_loss')   
plt.plot(AE_TestLoss, label='AE_TestLoss')    
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim(bottom =0 )
plt.title("MSE loss: train = {:.4f} , test = {:.4f}".format(MSE_loss[-1],AE_TestLoss[-1]))
plt.minorticks_on()
FigureDict.StoreFig(fig=FigureLoss1, name="Autoencoder_loss", saving=True)
plt.show()

##############################
lr_enc = HelpFunc.MovingAverage(Reporter.DUMPDICT['Lr_AE'],window=n_epoch)

Figlr = plt.figure(figsize=(10,6))
plt.plot(lr_enc, label='lr_AE')
plt.legend(loc=2)
plt.xlabel('Steps')
plt.ylabel('Learning Rate')
plt.minorticks_on()
FigureDict.StoreFig(fig=Figlr, name="Lr_vs_Epoch", saving=True)
plt.show()


#%% 11. Copying file and deleting all objects to clear memory

del FigureDict, ae, Reporter, ParameterFile
print(f"finished run please check   : {trainfolder} ")

