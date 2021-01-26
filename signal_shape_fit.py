# import ROOT in batch mode                                                                                                                                                          
import os
import sys
import PyFunctions
from PyFunctions import *
import math
from array import array
import re
import json
import types
import pandas as pd

#import sys                                                                                                                                                                          
oldargv = sys.argv[:]
sys.argv = [ '-b-' ]
import ROOT
ROOT.gROOT.SetBatch(True)
sys.argv = oldargv

import numpy as np
from array import array

# from ROOT import TH1F, TH1D, TH2D, TFile, TLorentzVector, TVector3, TChain, TProfile, TTree, TGraph                                                                                
from ROOT import *

# load FWLite C++ libraries                                                                                                                                                          
ROOT.gSystem.Load("libFWCoreFWLite.so");
ROOT.gSystem.Load("libDataFormatsFWLite.so");
ROOT.FWLiteEnabler.enable()

# load FWlite python libraries                                                                                                                                                       
from DataFormats.FWLite import Handle, Events

# masses = [0.5, 0.6, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 25]
masses = [3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 25]
# masses = [0.5, 0.6, 0.75, 1, 1.25, 1.5, 2]

# ctaus = [1,10,100]
ctaus = [10]

# mass = masses[int(sys.argv[1])]

tree_muMC = ROOT.TChain('t')
tree_muMC.Add("/cms/routray/hzd_mass_ctau_scan.root")

lxybins = np.array([[0.0,0.2], [0.2,1.0], [1.0,2.4], [2.4,3.1], [3.1,7.0], [7.0,11.0]])                             


def massfit(mass = 2,ctau = 1):

    x = ROOT.RooRealVar("x","x",float(mass-(100*binwidth)),float(mass+(100*binwidth)))
    l = ROOT.RooArgList(x)

    MC = ROOT.RooDataHist("signal", "signal", l, h)

    ################################dcbg#################################

    mean = ROOT.RooRealVar('mean', 'Mean of DoubleCB', float(mass- (25*binwidth)), float(mass + (25*binwidth)))
    sigma = ROOT.RooRealVar('sigma', 'Sigma of DoubleCB', 0,1)
    alpha_1 = ROOT.RooRealVar('alpha_1', 'alpha1 of DoubleCB',  1)
    alpha_2 = ROOT.RooRealVar('alpha_2', 'alpha2 of DoubleCB',  4)
    n_1 = ROOT.RooRealVar('n_1', 'n1 of DoubleCB', 2)
    n_2 = ROOT.RooRealVar('n_2', 'n2 of DoubleCB', 5)

    cbs_1 = ROOT.RooCBShape("CrystallBall_1", "CrystallBall_1", x, mean, sigma, alpha_1, n_1)
    cbs_2 = ROOT.RooCBShape("CrystallBall_2", "CrystallBall_2", x, mean, sigma, alpha_2, n_2)
    mc_frac = ROOT.RooRealVar('mc_frac', 'mc_frac', 0.45)

    mean1 = ROOT.RooRealVar("mean1","Mean of Gaussian",float(mass- (25*binwidth)),float(mass + (25*binwidth)))                
    sigma1 = ROOT.RooRealVar("sigma1","Width of Gaussian",0.35) #change this to 0.05 for mass < 2 GeV                                       
    gaus = ROOT.RooGaussian("gaus","gaus",x,mean1,sigma1)  
    mc_frac1 = ROOT.RooRealVar('mc_frac1', 'mc_frac1', 0.5)

    signal = ROOT.RooAddPdf('signal', 'signal', ROOT.RooArgList(cbs_1,cbs_2,gaus), ROOT.RooArgList(mc_frac,mc_frac1))

    nS = 10000000
    sig_norm = ROOT.RooRealVar("sig_norm","sig_norm",nS,0,10*nS)
    model = ROOT.RooAddPdf("model","model",ROOT.RooArgList(signal),ROOT.RooArgList(sig_norm))

    model.fitTo(MC)
    # signal.fitTo(MC)

    #########################################plot###################################

    c = ROOT.TCanvas("c","c")

    xframe = x.frame((ROOT.RooFit.Title("")))
    MC.plotOn(xframe, ROOT.RooFit.Name("MC"))
    model.plotOn(xframe,ROOT.RooFit.LineColor(3),ROOT.RooFit.Name("Gaussian"), ROOT.RooFit.LineStyle(2))
    # signal.plotOn(xframe,ROOT.RooFit.LineColor(4),ROOT.RooFit.Name("Gaussian"), ROOT.RooFit.LineStyle(2))
    chisq = xframe.chiSquare(2)
    print chisq
    # nll = result.minNll()
    
    xframe.GetXaxis().SetTitle("Dimuon Mass [GeV]")
    # xframe.SetTitle("mass {}GeV, ctau {}mm, #chi^{2}/ndf = {}".format(mass,ctau,chisq))
    # xframe.Draw()
    # c.Draw()
    
    c1 = ROOT.TCanvas("c1","c1")

    xframe1 = x.frame(ROOT.RooFit.Title('mass {}GeV, ctau {}mm, chi2/ndf = {:.2f}'.format(mass,ctau,chisq)))

    MC.plotOn(xframe1, ROOT.RooFit.Name("MC"))
    model.plotOn(xframe1,ROOT.RooFit.LineColor(3),ROOT.RooFit.Name("Gaussian"), ROOT.RooFit.LineStyle(2))
    # signal.plotOn(xframe,ROOT.RooFit.LineColor(4),ROOT.RooFit.Name("Gaussian"), ROOT.RooFit.LineStyle(2))                                                 
    xframe1.GetXaxis().SetTitle("Dimuon Mass [GeV]")
    xframe1.Draw()
    c1.Draw()
    c1.SaveAs("dCBsignalshape_hzd_vf/signal_shape_mass{}_ctau{}.png".format(mass,ctau))


    return (mean.getVal(), mean.getError(), mean1.getVal(), mean1.getError(), sigma.getVal(), sigma.getError(), mc_frac.getVal(), mc_frac.getError(), mc_frac1.getVal(), mc_frac1.getError())

    #######################################main############################################

for m in range(len(masses)):

    binwidth = 0.001*masses[m]

    for c in range(len(ctaus)):

        print "running on mass", masses[m], "and ctau", ctaus[c]

        h = ROOT.TH1F("h","h", int(round(50/binwidth)), 0, 50)
        tree_muMC.Draw('mass>>h','sample_mass == {} && sample_ctau == {}'.format(masses[m],ctaus[c]),'')
        fit = massfit(mass = masses[m], ctau = ctaus[c])

        arr = []
        arr = [masses[m], fit[0], fit[1], fit[2], fit[3], fit[4], fit[5], fit[6], fit[7], fit[8], fit[9]]
        df = pd.DataFrame([arr],columns=['mass','meandCB','meandCBerr','meanGaus','meanGauserr','sigma','sigmaerr','mc_fracdCB','mc_fracdCBerr','mcfracGaus','mcfracGauserr'])
        df.to_csv('dCBsignalshape_hzd_vf/dCBfithzd_mass{}_ctau{}_lxyall_vf.csv'.format(masses[m],ctaus[c]),index=False)


