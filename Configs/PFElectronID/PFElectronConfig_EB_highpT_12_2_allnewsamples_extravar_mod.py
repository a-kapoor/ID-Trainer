# In this file you can specify the training configuration
#####################################################################
######Do not touch this
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
#####################################################################
####Start here
#####################################################################
OutputDirName = 'PFElectronConfig_EB_highpT_12_2_allnewsamples_extravar' #All plots, models, config file will be stored here

Debug=False # If True, only a small subset of events/objects are used for either Signal or background #Useful for quick debugging

#Branches to read #Should be in the root files #Only the read branches can be later used for any purpose
branches=["scl_eta",
          "ele_pt",
          "matchedToGenEle",
          "matchedToGenPhoton",
          "matchedToGenTauJet",
          "matchedToHadron",
          "ele_convDist",
          "ele_convDcot",
          "EleMVACats",
          "ele_fbrem","ele_deltaetain", "ele_deltaphiin", "ele_oldsigmaietaieta", 
          "ele_oldhe", "ele_ep", "ele_olde15", "ele_eelepout",
          "ele_kfchi2", "ele_kfhits", "ele_expected_inner_hits","ele_dr03TkSumPt",
          "ele_dr03EcalRecHitSumEt","ele_dr03HcalTowerSumEt",
          "ele_ecalPFClusterIso","ele_hcalPFClusterIso",
          "ele_gsfchi2",
          'ele_conversionVertexFitProbability',
          "ele_nbrem",'ele_deltaetaseed','ele_hadronicOverEm','ele_olde25max','ele_olde55','passElectronSelection',
          'ele_oldcircularity','ele_oldsigmaiphiiphi','ele_oldr9',"scl_E","ele_SCfbrem","ele_IoEmIop","ele_psEoverEraw"]
#branches=["Electron_*"]
#Possible examples
# ["Electron_*","matchingflag",]
# ["Electron_pt", "Electron_deltaEtaSC", "Electron_r9","Electron_eta"]
# You need to read branches to use them anywhere

##### If True, this will save the dataframe as a csv and the next time you run the same training with different parameters, it will be much faster
SaveDataFrameCSV=True
##### If branches and files are same a "previous" (not this one) training and SaveDataFrameCSV was True, you can switch on loadfromsaved and it will be much quicker to run the this time
loadfromsaved=False

#pt and eta bins of interest -------------------------------------------------------------------
#will be used for robustness studies and will also be used for 2D pt-eta reweighing if the reweighing option is True
ptbins = [10,30,40,50,80,100,5000] 
etabins = [-1.6,-1.2,-0.8,-0.5,0.0,0.5,0.8,1.2,1.6]
ptwtvar='ele_pt'
etawtvar='scl_eta'
##pt and eta bins of interest -------------------------------------------------------------------

#Reweighting scheme -------------------------------------------------------------------
Reweighing = 'True' # This is independent of xsec reweighing (this reweighing will be done after taking into account xsec weight of multiple samples). Even if this is 'False', xsec reweighting will always be done.
WhichClassToReweightTo="NonIsolatedBackground" #2D pt-eta spectrum of all other classs will be reweighted to this class
#will only be used if Reweighing = 'True'
#Reweighting scheme -------------------------------------------------------------------

Classes = ['IsolatedSignal','NonIsolatedSignal','NonIsolatedBackground','FromHadronicTaus','FromPhotons'] 
ClassColors = ['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628']

#dictionary of processes
CommonSel='(ele_pt > 10) & (abs(scl_eta) < 1.442) & (abs(scl_eta) < 2.5)'

PromptSel='((matchedToGenEle == 1) | (matchedToGenEle == 2)) & (matchedToGenPhoton==0) & (index%8==0)'
bHadSel='(matchedToGenEle != 1) & (matchedToGenEle != 2) &  (matchedToHadron==3) & (matchedToGenTauJet==0) & (matchedToGenPhoton==0) & (index%3==0)'
QCDSel='(matchedToGenEle ==0) &  (matchedToHadron!=3) & (matchedToGenTauJet==0) & (matchedToGenPhoton==0) & (index%3==0)'
hadtauSel='(matchedToGenEle == 0) & (matchedToGenTauJet==1) & (matchedToGenPhoton==0) & (index%6==0)'
PhoSel='(matchedToGenEle != 1) & (matchedToGenEle != 2) &  (matchedToHadron==0) & (matchedToGenTauJet==0) & (matchedToGenPhoton==1) & (index%6==0)'

loc='/scratch/PFNtuples_July_correct/'
loc1='/scratch/'
import os
if 'cern.ch' in os.uname()[1]: loc='/eos/cms/store/group/phys_egamma/akapoor/ntuple_ForPFID_July_Correct/ntuple_PFID_July_correct/'

processes=[
    {
        'Class':'IsolatedSignal',
        'path':(loc1+'PFNtuples_CMSSW12/DYJets_incl_MLL-50_TuneCP5_14TeV-madgraphMLM-pythia8/crab_DYJets_incl_MLL-50_TuneCP5_14TeV-madgraphMLM-pythia8/211031_013727/0000/','.root'),
        'xsecwt': 1, #xsec wt if any, if none then it can be 1
        'selection':CommonSel+' & '+PromptSel, #selection for background
    },

    {
        'Class':'IsolatedSignal',
        'path':('/scratch/PFNtuples_CMSSW12/DYToEE_M-50_NNPDF31_TuneCP5_14TeV-powheg-pythia8/crab_DYToEE_M-50_NNPDF31_TuneCP5_14TeV-powheg-pythia8/211031_013735/0000/','.root'),
        'xsecwt': 1, #xsec wt if any, if none then it can be 1
        'selection':CommonSel+' & '+PromptSel, #selection for background
    },

    {
        'Class':'IsolatedSignal',
        'path':('/scratch/PFNtuples_CMSSW12/ZprimeToEE_M-3000_TuneCP5_14TeV-pythia8/crab_ZprimeToEE_M-3000_TuneCP5_14TeV-pythia8/211031_013712/0000/','.root'),
        'xsecwt': 1, #xsec wt if any, if none then it can be 1
        'selection':CommonSel+' & '+PromptSel, #selection for background
    },
    
    {
        'Class':'IsolatedSignal',
        'path':('/scratch/PFNtuples_CMSSW12/ZprimeToEE_M-4000_TuneCP5_14TeV-pythia8/crab_ZprimeToEE_M-4000_TuneCP5_14TeV-pythia8/211031_013720/0000/','.root'),
        'xsecwt': 1, #xsec wt if any, if none then it can be 1
        'selection':CommonSel+' & '+PromptSel, #selection for background
    },

    # {
    #     'Class':'NonIsolatedSignal',
    #     'path':[# loc+'mc/QCD_Pt_15to20_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210451/0000/output_1.root',
    #             # loc+'mc/QCD_Pt_15to20_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210451/0000/output_2.root',
    #             # loc+'mc/QCD_Pt_15to20_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210451/0000/output_3.root',
    #             # loc+'mc/QCD_Pt_15to20_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210451/0000/output_4.root',
    #             # loc+'mc/QCD_Pt_15to20_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210451/0000/output_5.root',
    #             # loc+'mc/QCD_Pt_15to20_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210451/0000/output_6.root',
    #             loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_1.root',
    #             loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_10.root',
    #             loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_11.root',
    #             loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_12.root',
    #             loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_13.root',
    #             loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_14.root',
    #             loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_2.root',
    #             loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_3.root',
    #             loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_4.root',
    #             loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_5.root',
    #             loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_6.root',
    #             loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_7.root',
    #             loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_8.root',
    #             loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_9.root',
    #             # loc+'mc/QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210503/0000/output_1.root',
    #             # loc+'mc/QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210503/0000/output_10.root',
    #             # loc+'mc/QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210503/0000/output_2.root',
    #             # loc+'mc/QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210503/0000/output_3.root',
    #             # loc+'mc/QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210503/0000/output_4.root',
    #             # loc+'mc/QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210503/0000/output_5.root',
    #             # loc+'mc/QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210503/0000/output_6.root',
    #             # loc+'mc/QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210503/0000/output_7.root',
    #             # loc+'mc/QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210503/0000/output_8.root',
    #             # loc+'mc/QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210503/0000/output_9.root',
    #             loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_1.root',
    #             loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_10.root',
    #             loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_11.root',
    #             loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_12.root',
    #             loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_13.root',
    #             loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_14.root',
    #             loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_15.root',
    #             loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_16.root',
    #             loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_17.root',
    #             loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_18.root',
    #             loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_2.root',
    #             loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_3.root',
    #             loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_4.root',
    #             loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_5.root',
    #             loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_6.root',
    #             loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_7.root',
    #             loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_8.root',
    #             loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_9.root',
    #             # loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_1.root',
    #             # loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_10.root',
    #             # loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_11.root',
    #             # loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_12.root',
    #             # loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_13.root',
    #             # loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_14.root',
    #             # loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_15.root',
    #             # loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_16.root',
    #             # loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_17.root',
    #             # loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_2.root',
    #             # loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_3.root',
    #             # loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_4.root',
    #             # loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_5.root',
    #             # loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_6.root',
    #             # loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_7.root',
    #             # loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_8.root',
    #             # loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_9.root',
    #             # loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_1.root',
    #             # loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_10.root',
    #             # loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_11.root',
    #             # loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_12.root',
    #             # loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_13.root',
    #             # loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_14.root',
    #             # loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_15.root',
    #             # loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_16.root',
    #             # loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_17.root',
    #             # loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_18.root',
    #             # loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_19.root',
    #             # loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_2.root',
    #             # loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_20.root',
    #             # loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_3.root',
    #             # loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_4.root',
    #             # loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_5.root',
    #             # loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_6.root',
    #             # loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_7.root',
    #             # loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_8.root',
    #             # loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_9.root',
    #             ],
    #     'xsecwt': 1, #xsec wt if any, if none then it can be 1
    #     'selection':CommonSel+' & '+bHadSel+'& (index%8==0)', #selection for background
    # },

    
    {
        'Class':'NonIsolatedSignal',
        'path':['/scratch/PFNtuples_CMSSW12/QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/211031_013847/0000/electron_ntuple_1.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/211031_013847/0000/electron_ntuple_10.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/211031_013847/0000/electron_ntuple_11.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/211031_013847/0000/electron_ntuple_12.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/211031_013847/0000/electron_ntuple_13.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/211031_013847/0000/electron_ntuple_14.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/211031_013847/0000/electron_ntuple_2.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/211031_013847/0000/electron_ntuple_3.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/211031_013847/0000/electron_ntuple_4.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/211031_013847/0000/electron_ntuple_5.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/211031_013847/0000/electron_ntuple_6.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/211031_013847/0000/electron_ntuple_7.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/211031_013847/0000/electron_ntuple_8.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_14TeV_pythia8/211031_013847/0000/electron_ntuple_9.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_170to250_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_14TeV_pythia8/211031_013916/0000/electron_ntuple_1.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_170to250_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_14TeV_pythia8/211031_013916/0000/electron_ntuple_10.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_170to250_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_14TeV_pythia8/211031_013916/0000/electron_ntuple_11.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_170to250_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_14TeV_pythia8/211031_013916/0000/electron_ntuple_12.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_170to250_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_14TeV_pythia8/211031_013916/0000/electron_ntuple_2.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_170to250_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_14TeV_pythia8/211031_013916/0000/electron_ntuple_3.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_170to250_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_14TeV_pythia8/211031_013916/0000/electron_ntuple_4.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_170to250_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_14TeV_pythia8/211031_013916/0000/electron_ntuple_5.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_170to250_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_14TeV_pythia8/211031_013916/0000/electron_ntuple_6.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_170to250_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_14TeV_pythia8/211031_013916/0000/electron_ntuple_7.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_170to250_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_14TeV_pythia8/211031_013916/0000/electron_ntuple_8.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_170to250_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_14TeV_pythia8/211031_013916/0000/electron_ntuple_9.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_20to30_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_14TeV_pythia8/211031_013855/0000/electron_ntuple_1.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_20to30_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_14TeV_pythia8/211031_013855/0000/electron_ntuple_10.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_20to30_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_14TeV_pythia8/211031_013855/0000/electron_ntuple_11.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_20to30_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_14TeV_pythia8/211031_013855/0000/electron_ntuple_2.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_20to30_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_14TeV_pythia8/211031_013855/0000/electron_ntuple_3.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_20to30_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_14TeV_pythia8/211031_013855/0000/electron_ntuple_4.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_20to30_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_14TeV_pythia8/211031_013855/0000/electron_ntuple_5.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_20to30_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_14TeV_pythia8/211031_013855/0000/electron_ntuple_6.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_20to30_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_14TeV_pythia8/211031_013855/0000/electron_ntuple_7.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_20to30_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_14TeV_pythia8/211031_013855/0000/electron_ntuple_8.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_20to30_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_14TeV_pythia8/211031_013855/0000/electron_ntuple_9.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_250toInf_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_14TeV_pythia8/211031_013923/0000/electron_ntuple_1.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_250toInf_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_14TeV_pythia8/211031_013923/0000/electron_ntuple_2.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_250toInf_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_14TeV_pythia8/211031_013923/0000/electron_ntuple_3.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_250toInf_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_14TeV_pythia8/211031_013923/0000/electron_ntuple_4.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_30to80_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_14TeV_pythia8/211031_013902/0000/electron_ntuple_1.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_30to80_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_14TeV_pythia8/211031_013902/0000/electron_ntuple_10.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_30to80_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_14TeV_pythia8/211031_013902/0000/electron_ntuple_11.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_30to80_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_14TeV_pythia8/211031_013902/0000/electron_ntuple_2.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_30to80_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_14TeV_pythia8/211031_013902/0000/electron_ntuple_3.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_30to80_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_14TeV_pythia8/211031_013902/0000/electron_ntuple_4.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_30to80_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_14TeV_pythia8/211031_013902/0000/electron_ntuple_5.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_30to80_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_14TeV_pythia8/211031_013902/0000/electron_ntuple_6.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_30to80_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_14TeV_pythia8/211031_013902/0000/electron_ntuple_7.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_30to80_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_14TeV_pythia8/211031_013902/0000/electron_ntuple_8.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_80to170_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_14TeV_pythia8_retry2/211101_132726/0000/electron_ntuple_1.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_80to170_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_14TeV_pythia8_retry2/211101_132726/0000/electron_ntuple_2.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_80to170_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_14TeV_pythia8_retry2/211101_132726/0000/electron_ntuple_3.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_80to170_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_14TeV_pythia8_retry2/211101_132726/0000/electron_ntuple_4.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_80to170_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_14TeV_pythia8_retry2/211101_132726/0000/electron_ntuple_5.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt_80to170_bcToE_TuneCP5_14TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_14TeV_pythia8_retry2/211101_132726/0000/electron_ntuple_6.root',],
        'xsecwt': 1, #xsec wt if any, if none then it can be 1
        'selection':CommonSel+' & '+bHadSel, #selection for background
    },

    # {
    #     'Class':'NonIsolatedBackground',
    #     'path':[loc+'mc/QCD_Pt-120to170_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-120to170_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog/210711_205519/0000/output_1.root',
    #             loc+'mc/QCD_Pt-170to300_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-170to300_EMEnriched_TuneCP5_14TeV_pythia8/210711_205641/0000/output_1.root',
    #             loc+'mc/QCD_Pt-30to50_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-30to50_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisknow/210711_205537/0000/output_1.root',
    #             loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_1.root',
    #             loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_10.root',
    #             loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_11.root',
    #             loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_12.root',
    #             loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_13.root',
    #             loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_2.root',
    #             loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_3.root',
    #             loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_4.root',
    #             loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_5.root',
    #             loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_6.root',
    #             loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_7.root',
    #             loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_8.root',
    #             loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_9.root',
    #             loc+'mc/QCD_Pt-80to120_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-80to120_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog/210711_205553/0000/output_1.root'],
    #     'xsecwt': 1, #xsec wt if any, if none then it can be 1
    #     'selection':CommonSel+' & '+QCDSel, #selection for background
    # },

    
    {
        'Class':'NonIsolatedBackground',
        'path':['/scratch/PFNtuples_CMSSW12/QCD_Pt-10to30_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-10to30_EMEnriched_TuneCP5_14TeV_pythia8/211031_013757/0000/electron_ntuple_1.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-10to30_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-10to30_EMEnriched_TuneCP5_14TeV_pythia8/211031_013757/0000/electron_ntuple_2.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-10to30_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-10to30_EMEnriched_TuneCP5_14TeV_pythia8/211031_013757/0000/electron_ntuple_3.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-10to30_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-10to30_EMEnriched_TuneCP5_14TeV_pythia8/211031_013757/0000/electron_ntuple_4.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-10to30_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-10to30_EMEnriched_TuneCP5_14TeV_pythia8/211031_013757/0000/electron_ntuple_5.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-10to30_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-10to30_EMEnriched_TuneCP5_14TeV_pythia8/211031_013757/0000/electron_ntuple_6.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-120to170_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-120to170_EMEnriched_TuneCP5_14TeV_pythia8/211031_013826/0000/electron_ntuple_1.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-120to170_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-120to170_EMEnriched_TuneCP5_14TeV_pythia8/211031_013826/0000/electron_ntuple_10.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-120to170_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-120to170_EMEnriched_TuneCP5_14TeV_pythia8/211031_013826/0000/electron_ntuple_11.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-120to170_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-120to170_EMEnriched_TuneCP5_14TeV_pythia8/211031_013826/0000/electron_ntuple_3.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-120to170_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-120to170_EMEnriched_TuneCP5_14TeV_pythia8/211031_013826/0000/electron_ntuple_4.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-120to170_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-120to170_EMEnriched_TuneCP5_14TeV_pythia8/211031_013826/0000/electron_ntuple_5.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-120to170_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-120to170_EMEnriched_TuneCP5_14TeV_pythia8/211031_013826/0000/electron_ntuple_6.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-120to170_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-120to170_EMEnriched_TuneCP5_14TeV_pythia8/211031_013826/0000/electron_ntuple_7.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-120to170_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-120to170_EMEnriched_TuneCP5_14TeV_pythia8/211031_013826/0000/electron_ntuple_8.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-120to170_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-120to170_EMEnriched_TuneCP5_14TeV_pythia8/211031_013826/0000/electron_ntuple_9.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-170to300_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-170to300_EMEnriched_TuneCP5_14TeV_pythia8/211031_013833/0000/electron_ntuple_1.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-170to300_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-170to300_EMEnriched_TuneCP5_14TeV_pythia8/211031_013833/0000/electron_ntuple_2.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-170to300_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-170to300_EMEnriched_TuneCP5_14TeV_pythia8/211031_013833/0000/electron_ntuple_3.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-170to300_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-170to300_EMEnriched_TuneCP5_14TeV_pythia8/211031_013833/0000/electron_ntuple_4.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-170to300_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-170to300_EMEnriched_TuneCP5_14TeV_pythia8/211031_013833/0000/electron_ntuple_5.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-170to300_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-170to300_EMEnriched_TuneCP5_14TeV_pythia8/211031_013833/0000/electron_ntuple_6.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-170to300_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-170to300_EMEnriched_TuneCP5_14TeV_pythia8/211031_013833/0000/electron_ntuple_7.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-170to300_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-170to300_EMEnriched_TuneCP5_14TeV_pythia8/211031_013833/0000/electron_ntuple_8.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-170to300_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-170to300_EMEnriched_TuneCP5_14TeV_pythia8/211031_013833/0000/electron_ntuple_9.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-300toInf_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-300toInf_EMEnriched_TuneCP5_14TeV_pythia8/211031_013840/0000/electron_ntuple_1.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-300toInf_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-300toInf_EMEnriched_TuneCP5_14TeV_pythia8/211031_013840/0000/electron_ntuple_10.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-300toInf_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-300toInf_EMEnriched_TuneCP5_14TeV_pythia8/211031_013840/0000/electron_ntuple_11.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-300toInf_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-300toInf_EMEnriched_TuneCP5_14TeV_pythia8/211031_013840/0000/electron_ntuple_2.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-300toInf_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-300toInf_EMEnriched_TuneCP5_14TeV_pythia8/211031_013840/0000/electron_ntuple_3.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-300toInf_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-300toInf_EMEnriched_TuneCP5_14TeV_pythia8/211031_013840/0000/electron_ntuple_4.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-300toInf_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-300toInf_EMEnriched_TuneCP5_14TeV_pythia8/211031_013840/0000/electron_ntuple_5.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-300toInf_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-300toInf_EMEnriched_TuneCP5_14TeV_pythia8/211031_013840/0000/electron_ntuple_6.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-300toInf_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-300toInf_EMEnriched_TuneCP5_14TeV_pythia8/211031_013840/0000/electron_ntuple_7.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-300toInf_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-300toInf_EMEnriched_TuneCP5_14TeV_pythia8/211031_013840/0000/electron_ntuple_8.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-300toInf_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-300toInf_EMEnriched_TuneCP5_14TeV_pythia8/211031_013840/0000/electron_ntuple_9.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-30to50_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-30to50_EMEnriched_TuneCP5_14TeV_pythia8/211031_013804/0000/electron_ntuple_1.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-30to50_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-30to50_EMEnriched_TuneCP5_14TeV_pythia8/211031_013804/0000/electron_ntuple_2.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-30to50_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-30to50_EMEnriched_TuneCP5_14TeV_pythia8/211031_013804/0000/electron_ntuple_3.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-30to50_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-30to50_EMEnriched_TuneCP5_14TeV_pythia8/211031_013804/0000/electron_ntuple_4.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/211031_013811/0000/electron_ntuple_1.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/211031_013811/0000/electron_ntuple_2.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/211031_013811/0000/electron_ntuple_3.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/211031_013811/0000/electron_ntuple_4.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/211031_013811/0000/electron_ntuple_5.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-80to120_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-80to120_EMEnriched_TuneCP5_14TeV_pythia8/211031_013819/0000/electron_ntuple_1.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-80to120_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-80to120_EMEnriched_TuneCP5_14TeV_pythia8/211031_013819/0000/electron_ntuple_2.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-80to120_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-80to120_EMEnriched_TuneCP5_14TeV_pythia8/211031_013819/0000/electron_ntuple_3.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-80to120_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-80to120_EMEnriched_TuneCP5_14TeV_pythia8/211031_013819/0000/electron_ntuple_4.root',
                '/scratch/PFNtuples_CMSSW12/QCD_Pt-80to120_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-80to120_EMEnriched_TuneCP5_14TeV_pythia8/211031_013819/0000/electron_ntuple_5.root',],
        'xsecwt': 1, #xsec wt if any, if none then it can be 1
        'selection':CommonSel+' & '+QCDSel, #selection for background
    },
    
    
    {
        'Class':'FromHadronicTaus',
        'path':['/scratch/PFNtuples_CMSSW12/TauGun_Pt-15to500_14TeV-pythia8/crab_TauGun_Pt-15to500_14TeV-pythia8/211031_013938/0000/electron_ntuple_1.root',
                '/scratch/PFNtuples_CMSSW12/TauGun_Pt-15to500_14TeV-pythia8/crab_TauGun_Pt-15to500_14TeV-pythia8/211031_013938/0000/electron_ntuple_2.root',
                '/scratch/PFNtuples_CMSSW12/TauGun_Pt-15to500_14TeV-pythia8/crab_TauGun_Pt-15to500_14TeV-pythia8/211031_013938/0000/electron_ntuple_3.root',
                '/scratch/PFNtuples_CMSSW12/TauGun_Pt-15to500_14TeV-pythia8/crab_TauGun_Pt-15to500_14TeV-pythia8/211031_013938/0000/electron_ntuple_4.root',
                '/scratch/PFNtuples_CMSSW12/TauGun_Pt-15to500_14TeV-pythia8/crab_TauGun_Pt-15to500_14TeV-pythia8/211031_013938/0000/electron_ntuple_5.root',
                '/scratch/PFNtuples_CMSSW12/TauGun_Pt-15to500_14TeV-pythia8/crab_TauGun_Pt-15to500_14TeV-pythia8/211031_013938/0000/electron_ntuple_6.root',
                '/scratch/PFNtuples_CMSSW12/TauGun_Pt-15to500_14TeV-pythia8/crab_TauGun_Pt-15to500_14TeV-pythia8/211031_013938/0000/electron_ntuple_7.root',
                '/scratch/PFNtuples_CMSSW12/TauGun_Pt-5to15_14TeV-pythia8/crab_TauGun_Pt-5to15_14TeV-pythia8/211031_013931/0000/electron_ntuple_1.root',
                '/scratch/PFNtuples_CMSSW12/TauGun_Pt-5to15_14TeV-pythia8/crab_TauGun_Pt-5to15_14TeV-pythia8/211031_013931/0000/electron_ntuple_2.root',
                '/scratch/PFNtuples_CMSSW12/TauGun_Pt-5to15_14TeV-pythia8/crab_TauGun_Pt-5to15_14TeV-pythia8/211031_013931/0000/electron_ntuple_3.root',
                '/scratch/PFNtuples_CMSSW12/TauGun_Pt-5to15_14TeV-pythia8/crab_TauGun_Pt-5to15_14TeV-pythia8/211031_013931/0000/electron_ntuple_4.root',
                '/scratch/PFNtuples_CMSSW12/TauGun_Pt-5to15_14TeV-pythia8/crab_TauGun_Pt-5to15_14TeV-pythia8/211031_013931/0000/electron_ntuple_5.root',
                '/scratch/PFNtuples_CMSSW12/TauGun_Pt-5to15_14TeV-pythia8/crab_TauGun_Pt-5to15_14TeV-pythia8/211031_013931/0000/electron_ntuple_6.root',
                ],
        'xsecwt': 1, #xsec wt if any, if none then it can be 1
        'selection':CommonSel+' & '+hadtauSel, #selection for background
    },

    {
        'Class':'FromPhotons',
        'path':['/scratch/PFNtuples_CMSSW12/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013743/0000/electron_ntuple_1.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013743/0000/electron_ntuple_10.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013743/0000/electron_ntuple_11.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013743/0000/electron_ntuple_12.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013743/0000/electron_ntuple_13.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013743/0000/electron_ntuple_14.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013743/0000/electron_ntuple_15.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013743/0000/electron_ntuple_16.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013743/0000/electron_ntuple_17.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013743/0000/electron_ntuple_18.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013743/0000/electron_ntuple_19.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013743/0000/electron_ntuple_2.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013743/0000/electron_ntuple_3.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013743/0000/electron_ntuple_4.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013743/0000/electron_ntuple_5.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013743/0000/electron_ntuple_6.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013743/0000/electron_ntuple_7.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013743/0000/electron_ntuple_8.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-10to40_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013743/0000/electron_ntuple_9.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013750/0000/electron_ntuple_1.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013750/0000/electron_ntuple_10.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013750/0000/electron_ntuple_11.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013750/0000/electron_ntuple_12.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013750/0000/electron_ntuple_13.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013750/0000/electron_ntuple_14.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013750/0000/electron_ntuple_15.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013750/0000/electron_ntuple_16.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013750/0000/electron_ntuple_17.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013750/0000/electron_ntuple_18.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013750/0000/electron_ntuple_19.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013750/0000/electron_ntuple_2.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013750/0000/electron_ntuple_3.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013750/0000/electron_ntuple_4.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013750/0000/electron_ntuple_5.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013750/0000/electron_ntuple_6.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013750/0000/electron_ntuple_7.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013750/0000/electron_ntuple_8.root',
                '/scratch/PFNtuples_CMSSW12/GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_TuneCP5_14TeV_Pythia8/211031_013750/0000/electron_ntuple_9.root',],
        'xsecwt': 1, #xsec wt if any, if none then it can be 1
        'selection':CommonSel+' & '+PhoSel, #selection for background
    },
]

#####################################################################

def modifydf(df):#Do not remove this function, even if empty
    print("Can be used to add new branches (The pandas dataframe style)")
    
    ############ Write you modifications inside this block #######
    #example:
    df["EBrem"]=df["scl_E"] * df["ele_SCfbrem"]
    #df["Electron_SCeta"]=df["Electron_deltaEtaSC"] + df["Electron_eta"]
    
    ####################################################
    
    return df

#####################################################################

Tree = "ntuplizer/tree"

#MVAs to use as a list of dictionaries
MVAs = [
    #can add as many as you like: For MVAtypes XGB and DNN are keywords, so names can be XGB_new, DNN_old etc. 
    #But keep XGB and DNN in the names (That is how the framework identifies which algo to run
    

      # {"MVAtype":"XGB_1", #Keyword to identify MVA method.
      #  "Color":"green", #Plot color for MVA
      #  "Label":"XGB try1", # label can be anything (this is how you will identify them on plot legends)
      #  "features":["ele_fbrem", "ele_deltaetain", "ele_deltaphiin", "ele_oldsigmaietaieta",
      #              "ele_oldhe", "ele_ep", "ele_olde15", "ele_eelepout",
      #              "ele_kfchi2", "ele_kfhits", "ele_expected_inner_hits","ele_dr03TkSumPt",
      #              "ele_dr03EcalRecHitSumEt","ele_dr03HcalTowerSumEt","ele_gsfchi2","scl_eta","ele_pt",
      #              "ele_ecalPFClusterIso","ele_hcalPFClusterIso",
      #              #'ele_conversionVertexFitProbability',
      #              'ele_nbrem','ele_deltaetaseed','ele_hadronicOverEm','ele_olde25max','ele_olde55'],#Input features #Should be branchs in your dataframe
      #  "featuresgettr":["fbrem",
      #                   "abs(deltaEtaSuperClusterTrackAtVtx)",
      #                   "abs(deltaPhiSuperClusterTrackAtVtx)",
      #                   "full5x5_sigmaIetaIeta",
      #                   "full5x5_hcalOverEcal",
      #                   "eSuperClusterOverP",
      #                   "full5x5_e1x5",
      #                   "eEleClusterOverPout",
      #                   "closestCtfTrackNormChi2",
      #                   "closestCtfTrackNLayers",
      #                   "gsfTrack.hitPattern.numberOfLostHits.MISSING_INNER_HITS",
      #                   "dr03TkSumPt",
      #                   "dr03EcalRecHitSumEt",
      #                   "dr03HcalTowerSumEt",
      #                   "gsfTrack.normalizedChi2",
      #                   "superCluster.eta",
      #                   "pt",
      #                   "ecalPFClusterIso",
      #                   "hcalPFClusterIso",
      #                   "numberOfBrems","abs(deltaEtaSeedClusterTrackAtCalo)","hadronicOverEm","full5x5_e2x5Max","full5x5_e5x5"],
      #  "feature_bins":[100 for i in range(24)],#same length as features
      #  #Binning used only for plotting features (should be in the same order as features), does not affect training
      #  'Scaler':"StandardScaler",
      #  'UseGPU':True, #If you have a GPU card, you can turn on this option (CUDA 10.0, Compute Capability 3.5 required)
      #  "XGBGridSearch":{'min_child_weight': [1, 5],
      #                 'gamma': [0.5, 1],
      #                 'subsample': [0.6, 0.8],
      #                 'colsample_bytree': [0.6, 0.8],
      #                 'max_depth': [3, 4]} #All standard XGB parameters supported
      #  },
    
    # {"MVAtype":"DNN_rechitiso_2drwt_withpteta",
    #  "Color":"green", #Plot color for MVA
    #  "Label":"DNN_rechitiso_2drwt_withpteta", # label can be anything (this is how you will identify them on plot legends)
    #  "features":["ele_fbrem", "ele_deltaetain", "ele_deltaphiin", "ele_oldsigmaietaieta", 
    #              "ele_oldhe", "ele_ep", "ele_olde15", "ele_eelepout",
    #              "ele_kfchi2", "ele_kfhits", "ele_expected_inner_hits","ele_dr03TkSumPt",
    #              "ele_dr03EcalRecHitSumEt","ele_dr03HcalTowerSumEt","ele_gsfchi2","scl_eta","ele_pt",
    #              #"ele_ecalPFClusterIso","ele_hcalPFClusterIso",
    #              #'ele_conversionVertexFitProbability',
    #              'ele_nbrem','ele_deltaetaseed','ele_hadronicOverEm','ele_olde25max','ele_olde55'],#Input features #Should be branchs in your dataframe
    #  "feature_bins":[100 for i in range(22)],#same length as features
    #  #Binning used only for plotting features (should be in the same order as features), does not affect training
    #  'Scaler':"MinMaxScaler",
    #  "DNNDict":{'epochs':1000, 'batchsize':5000, 'lr':0.001, 
    #             #The other parameters which are not here, can be modified in Trainer script
    #             'model': Sequential([Dense(24, kernel_initializer='glorot_normal', activation='relu'),
    #                                  Dense(48, activation="relu"),
    #                                  Dense(24, activation="relu"),
    #                                  Dropout(0.1),
    #                                  Dense(len(Classes),activation="softmax")]),
    #             'compile':{'loss':'categorical_crossentropy','optimizer':Adam(lr=0.001), 'metrics':['accuracy']},
    #             'earlyStopping': EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    #             #check the modelDNN1 function above, you can also create your own
    #            }
    # },

    #  {"MVAtype":"DNN_rechitandclusteriso_2drwt_withpteta_oldvariables",
    #  "Color":"green", #Plot color for MVA
    #  "Label":"DNN_rechitandclusteriso_2drwt_withpteta_oldvariables", # label can be anything (this is how you will identify them on plot legends)
    #  "features":["ele_fbrem", "ele_deltaetain", "ele_deltaphiin", "ele_oldsigmaietaieta",
    #              "ele_oldhe", "ele_ep", "ele_olde15", "ele_eelepout",
    #              "ele_kfchi2", "ele_kfhits", "ele_expected_inner_hits","ele_dr03TkSumPt",
    #              "ele_dr03EcalRecHitSumEt","ele_dr03HcalTowerSumEt","ele_gsfchi2","scl_eta","ele_pt",
    #              "ele_ecalPFClusterIso","ele_hcalPFClusterIso",
    #              #'ele_conversionVertexFitProbability',
    #              'ele_nbrem','ele_deltaetaseed','ele_hadronicOverEm','ele_olde25max','ele_olde55'],#Input features #Should be branchs in your dataframe
    #  "featuresgettr":["fbrem",
    #                   "abs(deltaEtaSuperClusterTrackAtVtx)",
    #                   "abs(deltaPhiSuperClusterTrackAtVtx)",
    #                   "full5x5_sigmaIetaIeta",
    #                   "full5x5_hcalOverEcal",
    #                   "eSuperClusterOverP",
    #                   "full5x5_e1x5",
    #                   "eEleClusterOverPout",
    #                   "closestCtfTrackNormChi2",
    #                   "closestCtfTrackNLayers",
    #                   "gsfTrack.hitPattern.numberOfLostHits.MISSING_INNER_HITS",
    #                   "dr03TkSumPt",
    #                   "dr03EcalRecHitSumEt",
    #                   "dr03HcalTowerSumEt",
    #                   "gsfTrack.normalizedChi2",
    #                   "superCluster.eta",
    #                   "pt",
    #                   "ecalPFClusterIso",
    #                   "hcalPFClusterIso",
    #                   "numberOfBrems","abs(deltaEtaSeedClusterTrackAtCalo)","hadronicOverEm","full5x5_e2x5Max","full5x5_e5x5"],
    #  "feature_bins":[100 for i in range(24)],#same length as features
    #  #Binning used only for plotting features (should be in the same order as features), does not affect training
    #  'Scaler':"StandardScaler",
    #  "DNNDict":{'epochs':100, 'batchsize':6000, 'lr':0.001,
    #             #The other parameters which are not here, can be modified in Trainer script
    #             'model': Sequential([Dense(24, kernel_initializer='glorot_normal', activation='relu',name='FirstLayer'),
    #                                  Dense(48, activation="relu"),
    #                                  Dense(24, activation="relu"),
    #                                  Dropout(0.1),
    #                                  Dense(len(Classes),activation="softmax",name='FinalLayer')]),
    #             'compile':{'loss':'categorical_crossentropy','optimizer':Adam(lr=0.001), 'metrics':['accuracy']},
    #             'earlyStopping': EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    #             #check the modelDNN1 function above, you can also create your own
    #            }
    # },

    
     {"MVAtype":"DNN_rechitandclusteriso_2drwt_withpteta_oldvariables_nopteta",
     "Color":"green", #Plot color for MVA
     "Label":"DNN_rechitandclusteriso_2drwt_withpteta_oldvariables_nopteta", # label can be anything (this is how you will identify them on plot legends)
     "features":["ele_fbrem", "ele_deltaetain", "ele_deltaphiin", "ele_oldsigmaietaieta",
                 "ele_oldhe", "ele_ep", "ele_olde15", "ele_eelepout",
                 "ele_kfchi2", "ele_kfhits", "ele_expected_inner_hits","ele_dr03TkSumPt",
                 #"ele_dr03EcalRecHitSumEt","ele_dr03HcalTowerSumEt",
                 "ele_gsfchi2",#"scl_eta","ele_pt",
                 #"ele_ecalPFClusterIso","ele_hcalPFClusterIso",
                 #'ele_conversionVertexFitProbability',
                 'ele_nbrem','ele_deltaetaseed','ele_hadronicOverEm','ele_olde25max','ele_olde55'],#Input features #Should be branchs in your dataframe
     "featuresgettr":["fbrem",
                      "abs(deltaEtaSuperClusterTrackAtVtx)",
                      "abs(deltaPhiSuperClusterTrackAtVtx)",
                      "full5x5_sigmaIetaIeta",
                      "full5x5_hcalOverEcal",
                      "eSuperClusterOverP",
                      "full5x5_e1x5",
                      "eEleClusterOverPout",
                      "closestCtfTrackNormChi2",
                      "closestCtfTrackNLayers",
                      "gsfTrack.hitPattern.numberOfLostHits.MISSING_INNER_HITS",
                      "dr03TkSumPt",
                      #"dr03EcalRecHitSumEt",
                      #"dr03HcalTowerSumEt",
                      "gsfTrack.normalizedChi2",
                      #"superCluster.eta",
                      #"pt",
                      #"ecalPFClusterIso",
                      #"hcalPFClusterIso",
                      "numberOfBrems","abs(deltaEtaSeedClusterTrackAtCalo)","hadronicOverEm","full5x5_e2x5Max","full5x5_e5x5"],
     "feature_bins":[100 for i in range(18)],#same length as features
     #Binning used only for plotting features (should be in the same order as features), does not affect training
     'Scaler':"StandardScaler",
     "DNNDict":{'epochs':100, 'batchsize':6000, 'lr':0.001,
                #The other parameters which are not here, can be modified in Trainer script
                'model': Sequential([Dense(18, kernel_initializer='glorot_normal', activation='relu',name='FirstLayer'),
                                     Dense(36, activation="relu"),
                                     Dense(18, activation="relu"),
                                     Dropout(0.1),
                                     Dense(len(Classes),activation="softmax",name='FinalLayer')]),
                'compile':{'loss':'categorical_crossentropy','optimizer':Adam(lr=0.001), 'metrics':['accuracy']},
                'earlyStopping': EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
                #check the modelDNN1 function above, you can also create your own
               }
    },
    
    # {"MVAtype":"DNN_clusteriso_2drwt_withpteta",
    #  "Color":"green", #Plot color for MVA
    #  "Label":"DNN_clusteriso_2drwt_withpteta", # label can be anything (this is how you will identify them on plot legends)
    #  "features":["ele_fbrem", "ele_deltaetain", "ele_deltaphiin", "ele_oldsigmaietaieta", 
    #              "ele_oldhe", "ele_ep", "ele_olde15", "ele_eelepout",
    #              "ele_kfchi2", "ele_kfhits", "ele_expected_inner_hits","ele_dr03TkSumPt",
    #              #"ele_dr03EcalRecHitSumEt","ele_dr03HcalTowerSumEt",
    #              "ele_gsfchi2",
    #              "scl_eta","ele_pt",
    #              "ele_ecalPFClusterIso","ele_hcalPFClusterIso",
    #              'ele_conversionVertexFitProbability',
    #              'ele_nbrem','ele_deltaetaseed','ele_hadronicOverEm','ele_olde25max','ele_olde55',
    #              'ele_oldcircularity','ele_oldsigmaiphiiphi','ele_oldr9',"scl_E","ele_SCfbrem","ele_IoEmIop","ele_psEoverEraw","EBrem"],#Input features #Should be branchs in your dataframe
    #  "featuresgettr":["fbrem",
    #                   "abs(deltaEtaSuperClusterTrackAtVtx)",
    #                   "abs(deltaPhiSuperClusterTrackAtVtx)",
    #                   "full5x5_sigmaIetaIeta",
    #                   "full5x5_hcalOverEcal",
    #                   "eSuperClusterOverP",
    #                   "full5x5_e1x5",
    #                   "eEleClusterOverPout",
    #                   "closestCtfTrackNormChi2", 
    #                   "closestCtfTrackNLayers",
    #                   "gsfTrack.hitPattern.numberOfLostHits.MISSING_INNER_HITS",
    #                   "dr03TkSumPt",
    #                   #"dr03EcalRecHitSumEt",
    #                   #"dr03HcalTowerSumEt",
    #                   "gsfTrack.normalizedChi2",
    #                   "superCluster.eta",
    #                   "pt",
    #                   "ecalPFClusterIso",
    #                   "hcalPFClusterIso",
    #                   "numberOfBrems","abs(deltaEtaSeedClusterTrackAtCalo)","hadronicOverEm","full5x5_e2x5Max","full5x5_e5x5",
    #                   "1.-full5x5_e1x5/full5x5_e5x5","full5x5_sigmaIphiIphi","full5x5_r9","superCluster.energy","superClusterFbrem",
    #                   "1.0/ecalEnergy-1.0/trackMomentumAtVtx.R","superCluster.preshowerEnergy/superCluster.rawEnergy","temp"],
    #  "feature_bins":[100 for i in range(31)],#same length as features
    #  #Binning used only for plotting features (should be in the same order as features), does not affect training
    #  'Scaler':"StandardScaler",
    #  "DNNDict":{'epochs':500, 'batchsize':3000, 'lr':0.001, 
    #             #The other parameters which are not here, can be modified in Trainer script
    #             'model': Sequential([Dense(31, kernel_initializer='glorot_normal', activation='relu',name='FirstLayer'),
    #                                  Dense(31, activation="relu"),
    #                                  Dense(31, activation="relu"),
    #                                  Dropout(0.1),
    #                                  Dense(len(Classes),activation="softmax",name='FinalLayer')]),
    #             'compile':{'loss':'categorical_crossentropy','optimizer':Adam(lr=0.001), 'metrics':['accuracy']},
    #             'earlyStopping': EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    #             #check the modelDNN1 function above, you can also create your own
    #            }
    # },

    
    # {"MVAtype":"DNN_rechitandclusteriso_2drwt_withpteta",
    #  "Color":"green", #Plot color for MVA
    #  "Label":"DNN_rechitandclusteriso_2drwt_withpteta", # label can be anything (this is how you will identify them on plot legends)
    #  "features":["ele_fbrem", "ele_deltaetain", "ele_deltaphiin", "ele_oldsigmaietaieta", 
    #              "ele_oldhe", "ele_ep", "ele_olde15", "ele_eelepout",
    #              "ele_kfchi2", "ele_kfhits", "ele_expected_inner_hits","ele_dr03TkSumPt",
    #              "ele_dr03EcalRecHitSumEt","ele_dr03HcalTowerSumEt","ele_gsfchi2","scl_eta","ele_pt",
    #              "ele_ecalPFClusterIso","ele_hcalPFClusterIso",
    #              'ele_conversionVertexFitProbability',
    #              'ele_nbrem','ele_deltaetaseed','ele_hadronicOverEm','ele_olde25max','ele_olde55',
    #              'ele_oldcircularity','ele_oldsigmaiphiiphi','ele_oldr9',"scl_E","ele_SCfbrem","ele_IoEmIop","ele_psEoverEraw","EBrem"],#Input features #Should be branchs in your dataframe
    #  "featuresgettr":["fbrem",
    #                   "abs(deltaEtaSuperClusterTrackAtVtx)",
    #                   "abs(deltaPhiSuperClusterTrackAtVtx)",
    #                   "full5x5_sigmaIetaIeta",
    #                   "full5x5_hcalOverEcal",
    #                   "eSuperClusterOverP",
    #                   "full5x5_e1x5",
    #                   "eEleClusterOverPout",
    #                   "closestCtfTrackNormChi2", 
    #                   "closestCtfTrackNLayers",
    #                   "gsfTrack.hitPattern.numberOfLostHits.MISSING_INNER_HITS",
    #                   "dr03TkSumPt",
    #                   "dr03EcalRecHitSumEt",
    #                   "dr03HcalTowerSumEt",
    #                   "gsfTrack.normalizedChi2",
    #                   "superCluster.eta",
    #                   "pt",
    #                   "ecalPFClusterIso",
    #                   "hcalPFClusterIso",
    #                   "numberOfBrems","abs(deltaEtaSeedClusterTrackAtCalo)","hadronicOverEm","full5x5_e2x5Max","full5x5_e5x5",
    #                   "1.-full5x5_e1x5/full5x5_e5x5","full5x5_sigmaIphiIphi","full5x5_r9","superCluster.energy","superClusterFbrem",
    #                   "1.0/ecalEnergy-1.0/trackMomentumAtVtx.R","superCluster.preshowerEnergy/superCluster.rawEnergy","temp"],
    #  "feature_bins":[100 for i in range(33)],#same length as features
    #  #Binning used only for plotting features (should be in the same order as features), does not affect training
    #  'Scaler':"StandardScaler",
    #  "DNNDict":{'epochs':500, 'batchsize':3000, 'lr':0.001, 
    #             #The other parameters which are not here, can be modified in Trainer script
    #             'model': Sequential([Dense(33, kernel_initializer='glorot_normal', activation='relu',name='FirstLayer'),
    #                                  Dense(33, activation="relu"),
    #                                  Dense(33, activation="relu"),
    #                                  Dropout(0.1),
    #                                  Dense(len(Classes),activation="softmax",name='FinalLayer')]),
    #             'compile':{'loss':'categorical_crossentropy','optimizer':Adam(lr=0.001), 'metrics':['accuracy']},
    #             'earlyStopping': EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    #             #check the modelDNN1 function above, you can also create your own
    #            }
    # },

    
    # {"MVAtype":"DNN_2drwt_withpteta",
    #  "Color":"green", #Plot color for MVA
    #  "Label":"DNN_rechitandclusteriso_2drwt_withpteta", # label can be anything (this is how you will identify them on plot legends)
    #  "features":["ele_fbrem", "ele_deltaetain", "ele_deltaphiin", "ele_oldsigmaietaieta", 
    #              "ele_oldhe", "ele_ep", "ele_olde15", "ele_eelepout",
    #              "ele_kfchi2", "ele_kfhits", "ele_expected_inner_hits","ele_dr03TkSumPt",
    #              #"ele_dr03EcalRecHitSumEt","ele_dr03HcalTowerSumEt",
    #              "ele_gsfchi2","scl_eta","ele_pt",
    #              #"ele_ecalPFClusterIso","ele_hcalPFClusterIso",
    #              'ele_conversionVertexFitProbability',
    #              'ele_nbrem','ele_deltaetaseed','ele_hadronicOverEm','ele_olde25max','ele_olde55',
    #              'ele_oldcircularity','ele_oldsigmaiphiiphi','ele_oldr9',"scl_E","ele_SCfbrem","ele_IoEmIop","ele_psEoverEraw","EBrem"],#Input features #Should be branchs in your dataframe
    #  "featuresgettr":["fbrem",
    #                   "abs(deltaEtaSuperClusterTrackAtVtx)",
    #                   "abs(deltaPhiSuperClusterTrackAtVtx)",
    #                   "full5x5_sigmaIetaIeta",
    #                   "full5x5_hcalOverEcal",
    #                   "eSuperClusterOverP",
    #                   "full5x5_e1x5",
    #                   "eEleClusterOverPout",
    #                   "closestCtfTrackNormChi2", 
    #                   "closestCtfTrackNLayers",
    #                   "gsfTrack.hitPattern.numberOfLostHits.MISSING_INNER_HITS",
    #                   "dr03TkSumPt",
    #                   #"dr03EcalRecHitSumEt",
    #                   #"dr03HcalTowerSumEt",
    #                   "gsfTrack.normalizedChi2",
    #                   "superCluster.eta",
    #                   "pt",
    #                   #"ecalPFClusterIso",
    #                   #"hcalPFClusterIso",
    #                   "numberOfBrems","abs(deltaEtaSeedClusterTrackAtCalo)","hadronicOverEm","full5x5_e2x5Max","full5x5_e5x5",
    #                   "1.-full5x5_e1x5/full5x5_e5x5","full5x5_sigmaIphiIphi","full5x5_r9","superCluster.energy","superClusterFbrem",
    #                   "1.0/ecalEnergy-1.0/trackMomentumAtVtx.R","superCluster.preshowerEnergy/superCluster.rawEnergy","temp"],
    #  "feature_bins":[100 for i in range(29)],#same length as features
    #  #Binning used only for plotting features (should be in the same order as features), does not affect training
    #  'Scaler':"StandardScaler",
    #  "DNNDict":{'epochs':500, 'batchsize':3000, 'lr':0.001, 
    #             #The other parameters which are not here, can be modified in Trainer script
    #             'model': Sequential([Dense(29, kernel_initializer='glorot_normal', activation='relu',name='FirstLayer'),
    #                                  Dense(29, activation="relu"),
    #                                  Dense(29, activation="relu"),
    #                                  Dropout(0.1),
    #                                  Dense(len(Classes),activation="softmax",name='FinalLayer')]),
    #             'compile':{'loss':'categorical_crossentropy','optimizer':Adam(lr=0.001), 'metrics':['accuracy']},
    #             'earlyStopping': EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    #             #check the modelDNN1 function above, you can also create your own
    #            }
    # },

]
################################

#binning for feature_bins can also be like this
# np.linspace(lower boundary, upper boundary, totalbins+1)
# example: np.linspace(0,20,21) 
# 20 bins from 0 to 20
#when not sure about the binning, you can just specify numbers, which will then correspond to total bins
#You can even specify lists like [10,20,30,100]

################################

#Working Point Flags to compare to (Should be in your ntuple and should also be read in branches)
OverlayWP=['passElectronSelection']
OverlayWPColors = ["black"] #Colors on plots for WPs

#To print thresholds of mva scores for corresponding signal efficiency
SigEffWPs=["98%","99%"] # Example for 80% and 90% Signal Efficiency Working Points
######### 


#####Optional Features

RandomState=42
#Choose the same number everytime for reproducibility

#MVAlogplot=False
#If true, MVA outputs are plotted in log scale

#Multicore=False
#If True all CPU cores available are used XGB 

testsize=0.2
#(0.2 means 20%) (How much data to use for testing)

#flatten=False
#For NanoAOD and other un-flattened trees, you can switch on this option to flatten branches with variable length for each event (Event level -> Object level)
#You can't flatten branches which have different length for the same events. For example: It is not possible to flatten electron and muon branches both at the same time, since each event could have different electrons vs muons. Branches that have only one value for each event, line Nelectrons, can certainly be read along with unflattened branches.

