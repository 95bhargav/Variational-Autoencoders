# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 21:21:18 2020

@author: Shukla
"""

#%% Importing Libraries
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import time
from Functions import imshow, PlotViz, one_hot_softmax
import HelperFunctions as HelpFunc

#%% Plot settings
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

#%% Parameter Initializations

version = '-ver_1.180--13-01-2021'
changes = "Using Encoder in conditional mode fully conditional for cifar"
dataset     = "cifar10"
EncoderType = 'vae'

global _DEVICE
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
fixed_samples       = 10
feature_length      = 200
num_classes         = 10
Mean                = (0.5,0.5,0.5)
Std                 = (0.5,0.5,0.5)
Batch_size          = 64 
n_epoch             = 100
Img_size            = 32
step_E              = 45
step_G              = 45
gammaE              = 0.1
gammaG              = 0.1


#%% Folder Creation


time_str=time.strftime("%y_%m_%d__%H_%M_%S"+version)
path=os.path.dirname(os.getcwd())
ver_nr  = version.split('-')[1]
datasets = dataset +'_'
folder  = datasets+ver_nr 
trainpath=os.path.join(path,"01_GAN_with_Encoder", folder)
trainfolder=os.path.join(trainpath,time_str)
if not os.path.exists(trainfolder):
    os.makedirs(trainfolder)                        

#%% Networks


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, modules =1):
        super(ResBlock, self).__init__()
        
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1,inplace=True))
        
        for i in range(modules):
            self.block.add_module(f"Sequence_{i}", nn.Sequential( 
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)))
            
            if i!= modules-1:
                self.block.add_module(f"Activation_{i}", nn.LeakyReLU(0.1, inplace= True))
        
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return nn.LeakyReLU(0.1,inplace=True)(self.block(x) + self.shortcut(x))

class ResNetEncoderVAE(nn.Module):
    def __init__(self,ResBlock, feature_size, mods=2, nef =16, img_size =32, labels = False):
        super(ResNetEncoderVAE,self).__init__()
        
        self.feature_size = feature_size
        self.image_size = img_size
        self.nef = nef
        self.mods = mods
        self.labels = 1 if labels else 0
        
        if self.labels:
            self.lin = nn.Sequential(
                nn.Linear(10, self.image_size), nn.ReLU(),
                nn.Linear(self.image_size, self.image_size*self.image_size),
                nn.ReLU())
        
        if self.image_size == 32:
            self.block  = nn.Sequential(
                    nn.Conv2d(3+ self.labels, self.nef, 1, 1, 0, bias= False),
                    nn.BatchNorm2d(self.nef),
                    nn.LeakyReLU(0.2, inplace= True)) 
            
        elif self.image_size ==64:
            self.block  = nn.Sequential(
                    nn.Conv2d(3+ self.labels, self.nef, 4, 2, 1, bias= False),
                    nn.BatchNorm2d(self.nef),
                    nn.LeakyReLU(0.2, inplace= True))
            
        self.block1 = ResBlock(self.nef,     self.nef*2, 1, 1)
        self.block2 = ResBlock(self.nef*2,   self.nef*4, 2, self.mods )
        self.block3 = ResBlock(self.nef*4,   self.nef*4, 2, self.mods )
        self.block4 = ResBlock(self.nef*4,   self.nef*8, 2, self.mods )
        self.final  = nn.Sequential(
                        nn.Conv2d(self.nef*8,self.feature_size*2,4,1,0),
                        nn.Tanh())
        self.mu     = nn.Linear(self.feature_size*2, self.feature_size)
        self.logvar = nn.Linear(self.feature_size*2, self.feature_size)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = mu+ (eps*std)
        return sample
    
    
    def forward(self,image, labels = None):
        
        if self.labels:
            lab = self.lin(labels)
            lab = lab.reshape(-1,1, self.image_size,self.image_size)
            cat = torch.cat((image,lab),1)
        else:
            cat = image
        
        x = self.block(cat)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        feat= self.final(x)
        feat            = feat.reshape(-1,self.feature_size*2)
        mu              = self.mu(feat)
        log_var         = self.logvar(feat)
        sampled_feat    = self.reparameterize(mu,log_var)
        sampled_feat    = nn.Tanh()(sampled_feat)
        return  sampled_feat, mu, log_var
    
class generator2(nn.Module):
    def __init__(self, In, Out, feature_size, label_size=10, image_size = 32):
        super(generator2,self).__init__()
        
        self.In = In
        self.Out= Out
        self.feature_size = feature_size
        self.label_size = label_size
        self.image_size =image_size
        self.main = nn.Sequential(
                        #feat+labelsx1x1
                        nn.ConvTranspose2d(self.feature_size+self.label_size, Out,
                                           5,1,0, bias= False),
                        nn.BatchNorm2d(Out),
                        nn.LeakyReLU(0.2,inplace=True),
                        #512x5x5
                        nn.ConvTranspose2d(Out, Out//2, 3,1,1, bias= False),
                        nn.BatchNorm2d(Out//2),
                        nn.LeakyReLU(0.2,inplace=True),
                        #256x5x5
                        nn.ConvTranspose2d(Out//2, Out//2, 4,2,0, bias= False),
                        nn.BatchNorm2d(Out//2),
                        nn.LeakyReLU(0.2,inplace=True),
                        #256x12x12
                        nn.ConvTranspose2d(Out//2, Out//4, 3,1,0, bias= False),
                        nn.BatchNorm2d(Out//4),
                        nn.LeakyReLU(0.2,inplace=True),
                        #128x14x14
                        nn.ConvTranspose2d(Out//4, Out//4, 3,1,0, bias= False),
                        nn.BatchNorm2d(Out//4),
                        nn.LeakyReLU(0.2,inplace=True),
                        #128x16x16
                        nn.ConvTranspose2d(Out//4, Out//8, 4,2,1, bias= False),
                        nn.BatchNorm2d(Out//8),
                        nn.LeakyReLU(0.2,inplace=True),
                        #64x32x32
                        nn.ConvTranspose2d(Out//8, Out//16, 3,1,1, bias= False),
                        nn.BatchNorm2d(Out//16),
                        nn.LeakyReLU(0.2,inplace=True),
                        #32x32x32
                        nn.ConvTranspose2d(Out//16, Out//32, 3,1,1, bias= False),
                        nn.BatchNorm2d(Out//32),
                        nn.LeakyReLU(0.2,inplace=True))
                        #16x32x32
        if self.image_size ==32:
            self.main.add_module("Final", nn.Sequential(nn.ConvTranspose2d(Out//32, In, 3, 1, 1, bias=False),
                                                        nn.Tanh()))#3x32x32
        elif self.image_size ==64:
            self.main.add_module("Final", nn.Sequential(
                nn.ConvTranspose2d(Out//32, In, 4, 2, 1 ,bias = False),
                nn.Tanh()))

    def forward(self, features, labels, mode = "normal" ):
        
        if mode == "normal":
            labels = nn.Softmax(dim = -1)(labels)
        elif mode == "one_hot":
            labels = labels
        concat = torch.cat((features,labels),-1)
        concat = concat.reshape(-1,self.feature_size+self.label_size,1,1)
        gen_image = self.main(concat)
        return gen_image
          
#%% Training loop and testing loop

def test(enc, Generator,  fixed_data, criterionMSE, epoch):
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
        enc.eval()
        Generator.eval()
        for i,(image,label) in enumerate(fixed_data):
            image = image.to(_DEVICE)
            label = label.to(_DEVICE)
            
            one_hot_labels = torch.nn.functional.one_hot(label, num_classes).float()
            # feat,mu,logvar  = enc(image, labels =one_hot_labels)
            feat  = enc(image, labels =one_hot_labels)
            test_img       = Generator(feat,one_hot_labels, mode= "one_hot")
            mse_loss = criterionMSE( test_img, image)
            
            # loss = VAE_LOSS( test_img, image, mu, logvar)
            
            ae_loss += mse_loss.item()
    loss_ae                 =  (ae_loss/len(fixed_data))
    
    return loss_ae
       
def train(CurrentBatch, epoch, Encoder, Generator, optE, optG,
          criterionMSE, train_data, fixed_data, trainfolder,
          Reporter):
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
    
    
    Encoder.train()
    Generator.train()
    for i, (image,label) in enumerate(train_data):
        image = image.to(_DEVICE)
        label = label.to(_DEVICE)
        
        
        ## Traditional Autoencoder training loop
        optE.zero_grad()
        optG.zero_grad()
        
        one_hot_labels = torch.nn.functional.one_hot(label, num_classes).float()
        features = Encoder(image,labels= one_hot_labels)   #, mu, log_var
        recon_img   = Generator(features,one_hot_labels,mode= "one_hot")
        
        # loss = VAE_LOSS(recon_img,image, mu, log_var)
        
        loss = criterionMSE(recon_img,image)
        loss.backward()
        
        optE.step()
        optG.step()
        
        # Testing loop built-in
        if CurrentBatch % len(train_data) == 0:
            loss_ae_test= test(Encoder, Generator, fixed_data, criterionMSE, epoch)
                
        ## Plotting reconstructions
            with torch.no_grad():
                imag, label = next(iter(train_data))
                imag= imag.to(_DEVICE)
                label = label.to(_DEVICE)
                one_hot_labels = torch.nn.functional.one_hot(label, num_classes).float()
                feat = Encoder(imag,labels= one_hot_labels) #,_,_ 
                recon  = Generator(feat,one_hot_labels,mode= "one_hot")
                
                img_title_list = ["Real","Recon"]
                PlotTitle = "train_epoch_"+str(epoch)
                FigureDict = HelpFunc.FigureDict(os.path.join(trainfolder,"train_plots"),dpi =300 )
                PlotViz(imag, recon, img_title_list, FigureDict, 
                        PlotTitle, "cifar10")
            
                Encoder.eval()
                Generator.eval()
                
                image,label = next(iter(fixed_data))
                image = image.to(_DEVICE)
                label = label.to(_DEVICE)
                one_hot_labels = torch.nn.functional.one_hot(label, num_classes).float()
                feat        = Encoder(image,labels= one_hot_labels)#,_,_
                recon_image = Generator(feat,one_hot_labels, mode= "one_hot")
                
                img_title_list = ["Real","Recon"]
                PlotTitle = "test_epoch_"+str(epoch)
                FigureDict = HelpFunc.FigureDict(os.path.join(trainfolder,"test_plots"),dpi =300) 
                PlotViz(image, recon_image, img_title_list, FigureDict, 
                        PlotTitle, "cifar10")
                
                ## Reconstructions on fixed Noise
                fixed_image = Generator(fixed_noise,fixed_labels, mode ="one_hot")
                FigureDict = HelpFunc.FigureDict(os.path.join(trainfolder,"noise_recn"),dpi =300 )
                fig_ae  = plt.figure(figsize=(15,15),dpi=300)
                if fig_ae is not None:
                    fig_ae.suptitle("Reconstructions on Fixed noise and labels")
                
                for i in range(100):
                    ax=plt.subplot(10,10,i+1)
                    imshow(fixed_image[i],'cifar10')
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plt.xticks([])
                    plt.yticks([])
                    ax.set_title("{}".format(class_labels[torch.argmax(fixed_labels[i])]) )
                FigureDict.StoreFig(fig=fig_ae, name=PlotTitle,saving=True)
                plt.show()
        
        ## Reporting   
        Reporter.DUMPDICT['Lr_Enc'].append(optE.param_groups[0]['lr'])
        Reporter.DUMPDICT['Lr_Gen'].append(optG.param_groups[0]['lr'])
        
        Reporter.SetValues([epoch+1, CurrentBatch,
                            (time.time()-Reporter.DUMPDICT["starttime"])/60,
                            loss.item(), loss_ae_test])
        CurrentBatch+=1
    return CurrentBatch


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.mse_loss = nn.MSELoss(size_average=False)

    def forward(self, recon_x, x, mu, logvar):
        MSE = self.mse_loss(recon_x, x)

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE + KLD
VAE_LOSS = Loss()




#%%
data_path = os.path.dirname(path)
if dataset == "cifar10":
    trainloader,testloader = HelpFunc.LoadCifar10(path =os.path.join(data_path, "Datasets"),
                                              minibatch = Batch_size,
                                              normalization = "-11",
                                              image_size = Img_size)
    class_labels    = ["airplane",'automobile','bird','cat','deer','dog','frog','horse','ship','truck']
elif dataset == "stl10":
    trainloader,testloader = HelpFunc.LoadSTL10(path =os.path.join(data_path, "Datasets"),
                                                minibatch = Batch_size,
                                                normalization ="-11",
                                                train_split ='train',
                                                image_size = Img_size)
    class_labels= ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
print('Loading Data.....')
#%%
fixed_labels    = one_hot_softmax(1, samples = fixed_samples, classes =num_classes)      
fixed_labels    = torch.cat(([fixed_labels[i] for i in range(len(fixed_labels))])).to(_DEVICE)
fixed_noise     = ((torch.rand(num_classes*fixed_samples, feature_length)-0.5)*2).to(_DEVICE)  

criterionMSE    = nn.MSELoss()
        
if dataset == "cifar10":
    
    enc             = ResNetEncoderVAE(ResBlock,feature_size= feature_length, mods = 2,\
                                    nef= 54, img_size= Img_size, labels= True).to(_DEVICE)
    gen = generator2(In =3,Out = 323 ,feature_size= feature_length,\
                     label_size= num_classes, image_size = Img_size).to(_DEVICE)

    optE = torch.optim.Adam(enc.parameters(), lr = 0.001)#0.000363
    optG = torch.optim.RMSprop(gen.parameters(), lr = 0.001  )#0.000498
    
elif  dataset == "stl10":
    enc             = ResNetEncoderVAE(ResBlock,feature_size= feature_length, mods = 1,\
                                    nef= 56, img_size= Img_size, labels= True).to(_DEVICE)
    gen = generator2(In =3,Out = 304 ,feature_size= feature_length,\
                     label_size= num_classes, image_size = Img_size).to(_DEVICE)

    optE = torch.optim.RMSprop(enc.parameters(), lr =  0.000148)
    optG = torch.optim.Adam(gen.parameters(), lr = 0.002519)
 
print("\n",gen,"\n")
print("\n",enc,"\n")
schedE = torch.optim.lr_scheduler.StepLR(optE, step_size=step_E, gamma=gammaE)
schedG = torch.optim.lr_scheduler.StepLR(optG, step_size=step_G, gamma=gammaG)
print("\n Changes : ", changes, "\n ")
print("\n Starting Training Process......")   
#%% Training


ReporterNames=['Epoch',"Batch", "Time","Mse_loss", "AE_TestLoss"]

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
Reporter.DUMPDICT['Lr_Enc']         =   []
Reporter.DUMPDICT['Lr_Gen']         =   []

epoch = 0
for epoch in range(n_epoch):
    CurrentBatch=train(CurrentBatch, epoch, enc, gen, optE, optG, criterionMSE,
                   trainloader, testloader ,trainfolder, Reporter)
    schedE.step()
    schedG.step()
    
    
#%% Plotting 

##############################   
FigureDict = HelpFunc.FigureDict(os.path.join(trainfolder,"AE_metrics_Plots"),dpi =300 )
loss= HelpFunc.MovingAverage(Reporter.VALS['Mse_loss'],window=n_epoch)
AE_TestLoss= HelpFunc.MovingAverage(Reporter.VALS['AE_TestLoss'],window=n_epoch)      


FigureLoss1=plt.figure(figsize=(8,5))
plt.plot(loss, label = "MSE_loss")  
plt.plot(AE_TestLoss, label='AE_TestLoss')    
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim(bottom =0 )
plt.title("MSE loss: train = {:.5f} , test = {:.5f}".format(loss[-1],AE_TestLoss[-1]))

plt.minorticks_on()
FigureDict.StoreFig(fig=FigureLoss1, name="Autoencoder_loss", saving=True)
plt.show()

##############################
lr_enc = HelpFunc.MovingAverage(Reporter.DUMPDICT['Lr_Enc'],window=n_epoch)
lr_gen = HelpFunc.MovingAverage(Reporter.DUMPDICT['Lr_Gen'],window=n_epoch)
Figlr = plt.figure(figsize=(10,6))
plt.plot(lr_enc, label='lr_enc')
plt.plot(lr_gen, label='lr_gen')
plt.legend(loc=2)
plt.xlabel('Steps')
plt.ylabel('Learning Rate')
plt.yscale('log')

plt.minorticks_on()
FigureDict.StoreFig(fig=Figlr, name="Lr_vs_Epoch", saving=True)
plt.show()


#%%

print("finished run please check : {} ".format(trainfolder))

