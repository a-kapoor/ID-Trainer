universe = vanilla
+JobFlavour = "workday"
executable            = train.sh
#arguments = "Trainer_cmsml_exp_with3rdnodecut.py Configs/PFElectronID/PFElectronConfig_lowpT_12_2_allnewsamples_3rdnodecut"
log = test.log
output = condor_ouput/outfile.$(Cluster).$(Process).out
error = condor_ouput/errors.$(Cluster).$(Process).err
request_GPUs = 1
request_CPUs = 8
+testJob = True
queue 
