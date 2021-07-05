# In this file you can specify the training configuration

#####################################################################
######Do not touch this
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
#####################################################################
####Start here
#####################################################################
OutputDirName = 'Output' #All plots, models, config file will be stored here

Debug=True # If True, only a very small subset of events/objects are used for either Signal or background

branches=["Electron_*"]

##### For NanoAOD and other un-flattened trees, you can switch on this option to flatten branches with variable lenght for each event (Event level -> Object level)
flatten=True

##### If True, this will save the datafrme as a csv and the next time you run the same training with different parameters, it will be much faster
SaveDataFrameCSV=True
##### If branches and files are same a "previous" (not this one) training and SaveDataFrameCSV was True, you can switch on loadfromsaved and it will be much quicker to run the this time
loadfromsaved=False

#Common cuts for both signal and background (Would generally correspond to the training region)
CommonCut = "(Electron_pt>10) & (abs(Electron_eta)<1.566)" 
#This is barrel and pt>10 GeV
#endcap would be
#(ele_pt > 10) & (abs(scl_eta)>1.566)

Classes = ['Background','Signal'] #Which classes are you going to use (Do not modify this if you don't have a reason to)
## YES MULTICLASS IN COMING SOON

#dictionary of processes
processes=[
    {
        'Class':'Background',
        'path':'DYJetsToLL_M-50_v7_ElePromptGenMatched.root', #path of root file
        'xsecwt': 1, #xsec wt if any, if none then it can be 1
        'CommonSelection':CommonCut, #Common selection for all classes
        'selection':"(Electron_promptgenmatched == 0)", #selection for background
    },
    
    {
        'Class':'Signal',
        'path':'DYJetsToLL_M-50_v7_ElePromptGenMatched.root', #path of root file
        'xsecwt': 1, #xsec wt if any, if none then it can be 1
        'CommonSelection':CommonCut, #Common selection for all classes
        'selection':"(Electron_promptgenmatched == 1)", #selection for signal
        }
]

#####################################################################

testsize=0.2 #(0.2 means 20%)

def modfiydf(df):#Do not remove this function, even if empty
    #Can be used to add new branches (The pandas dataframe style)
    
    ############ Write you modifications inside this block #######
    #example:
    #df["Electron_SCeta"]=df["Electron_deltaEtaSC"] + df["Electron_eta"]
    
    ####################################################
    
    return 0

Tree = "Events" #Location/Name of tree inside Root files

################################

#MVAs to use as a list, e.g : ["XGB","DNN", "Genetic"]
MVAs = ["XGB_1","DNN_1"] 
#XGB and DNN are keywords so names can be XGB_new, DNN_old etc. But keep XGB and DNN in the names (That is how the framework identifies which algo to run

MVAColors = ["green","blue"] #Plot colors for MVAs

MVALabels = {"XGB_1" : "XGB masscut",
             "DNN_1" : "DNN masscut"
            } #These labels can be anything (this is how you will identify them on plot legends)

################################
features = {
            "XGB_1":["Electron_pt", "Electron_deltaEtaSC", "Electron_r9"],
            "DNN_1":["Electron_pt", "Electron_deltaEtaSC", "Electron_r9"]
           } #Input features to MVA #Should be in your ntuples

feature_bins = {
                "XGB_1":[100, 100, 100],
                "DNN_1":[100, 100, 100]
               } #Binning used only for plotting features (should be in the same order as features), does not affect training
#template 
#np.linspace(lower boundary, upper boundary, totalbins+1)

#when not sure about the binning, you can just specify numbers, which will then correspond to total bins
#You can even specify lists like [10,20,30,100]

#EGamma WPs to compare to (Should be in your ntuple)
OverlayWP=["Electron_mvaFall17V2noIso_WP90"]
OverlayWPColors = ["black","purple"] #Colors on plots for WPs
#####################################################################

######### Grid Search parameters for XGB (will only be used if MVAs contains "XGB"
XGBGridSearch= {
                "XGB_1": {'learning_rate':[0.1, 0.01, 0.001]}
               }

Scaler = {"XGB_1":"MinMaxScaler",
          "DNN_1":"MinMaxScaler" }
#
#To choose just one value for a parameter you can just specify value but in a list 
#Like "XGB_1":{'gamma':[0.5],'learning_rate':[0.1, 0.01]} 
#Here gamma is fixed but learning_rate will be varied

#The above are just one/two paramter grids
#All other parameters are XGB default values
#But you give any number you want:
#example:
#XGBGridSearch= {'learning_rate':[0.1, 0.01, 0.001],      
#                'min_child_weight': [1, 5, 10],
#                'gamma': [0.5, 1, 1.5, 2, 5],
#                'subsample': [0.6, 0.8, 1.0],
#                'colsample_bytree': [0.6, 0.8, 1.0],
#                'max_depth': [3, 4, 5]}
#Just rememeber the larger the grid the more time optimization takes

#Example for DNN_1
modelDNN_DNN_1=Sequential()
modelDNN_DNN_1.add(Dense(2*len(features["DNN_1"]), kernel_initializer='glorot_normal', activation='relu', input_dim=len(features["DNN_1"])))
modelDNN_DNN_1.add(Dense(len(features["DNN_1"]), kernel_initializer='glorot_normal', activation='relu'))
modelDNN_DNN_1.add(Dropout(0.1))
modelDNN_DNN_1.add(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))
DNNDict={
         "DNN_1":{'epochs':10, 'batchsize':100, 'lr':0.001, 'model':modelDNN_DNN_1}
        }


#####################################################################

SigEffWPs=["80%","90%"] # Example for 80% and 90% Signal Efficiency Working Points

######### Reweighting scheme #Feature not available but planned
Reweighing = 'ptetaSig'
ptbins = [10,30,40,50,100,5000] 
etabins = [-1.6,-1.0,0.0,1.0,1.6]
ptwtvar='Electron_pt'
etawtvar='Electron_eta'

'''
Possible choices :
Nothing : No reweighting
ptetaSig : To Signal pt-eta spectrum 
ptetaBkg : To Background pt-eta spectrum
'''

#####Optional Features
#RandomState=42 # Choose the same number everytime for reproducibility
#MVAlogplot=False #If true, MVA outputs are plotted in log scale
#Multicore=True #If True all CPU cores available are used XGB 