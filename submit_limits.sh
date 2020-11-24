#!/bin/bash                                                                                                                         

### NEED to get CMSSW software working with condor                                                                                 
 
#source /osg/osg3.2/osg-wn-client/setup.sh                                                                                         
 
export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch
source $VO_CMS_SW_DIR/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc630
export CMSSW_GIT_REFERENCE=/cvmfs/cms.cern.ch/cmssw.git
cd /cms/routray/combinetest/CMSSW_10_2_13/src/
eval `scramv1 runtime -sh`
export HOME=/users/h2/routray/
###                                                                                                                                 

cd HiggsAnalysis/CombinedLimit/fitsandlimits

echo $1

python cal_limits_v0.py $1 
