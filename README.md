step1>>

export SCRAM_ARCH=slc7_amd64_gcc700\
cmsrel CMSSW_10_2_13\
cd CMSSW_10_2_13/src\
cmsenv\
git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit\
cd HiggsAnalysis/CombinedLimit\

step2>> 

cd $CMSSW_BASE/src/HiggsAnalysis/CombinedLimit\
git fetch origin\
git checkout v8.1.0\
scramv1 b clean; scramv1 b # always make a clean build\

step3>>

git clone git@github.com:hardikroutray/displacedscouting_fitscountlimits.git\

step4>>

Create singlebin(fit and count) datacards for a particular mass(masses array inside the make_datacard_v0 script)\
python make_datacard_v0.py $1 will create a mass folder with all datacards alongwith bias tests as well as goodness of fit tests\

step5>>

Create multibin datacards for the same mass and store Nevt UL or xsec UL or br UL as csv file\
python cal_limit_v0.py $1 will create multibin datacards for given lifetime(lifetimes array inside cal_limit_v0 script) inside the created mass folders and store limits for each mass and various lifetimes as a csv file

step5>>

python csvanalyzer.py combines the csvfiles to create a mass-lifetime UL csvfile\

step6>>

Submit separate jobs for each mass using jcl scripts. update masses array in make_datacard_v0.py and cal_limit_v0.py.\
condor_submit submit_fitter.jcl\
condor_submit submit_limits.jcl\

Miscellanous>>

acc_fit_vf.py does spline fit of the acceptances for both MC. BPhi acceptances needs to be updated.\
systematics to be updated make_datacard_v0.py\
root6 not compatible with higgs combine environment and needs patches to CMSSW. should not be necessary here\
