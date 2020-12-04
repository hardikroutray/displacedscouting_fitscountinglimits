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


def num_after_point(x):
    s = str(x)
    if not '.' in s:
        return 0
    return len(s) - s.index('.') - 1

# masses = [0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7,0.725,0.75,0.775,0.8,0.825,0.85,0.875,0.9,0.925,0.95,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.25,4.5,4.75,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

# masses = [8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

masses = [8,9]

print len(masses)

mass = masses[int(sys.argv[1])]

print "running on mass", mass

if not os.path.exists("mass_{}".format(mass)):
                        os.makedirs("mass_{}".format(mass))

os.chdir("./mass_{}".format(mass))

# tree_muMC = ROOT.TChain('t')
# tree_muMC.Add("/cms/routray/hzd_mass_ctau_scan.root")
tree_mudata = ROOT.TChain('t')
tree_mudata.Add("/cms/routray/data_subset_pass_all.root")
# tree_mudata.Print()

lxybins = np.array([[0.0,0.2], [0.2,1.0], [1.0,2.4], [2.4,3.1], [3.1,7.0], [7.0,11.0]])
# lxybins = np.array([[0.04,0.2]])
#print lxybins[0,0], lxybins[0,1]

mu = mass
sig = 0.01*mass
gam = 0.01*mass

binwidth = sig/10
ndecimal = num_after_point(binwidth) + 1  

# print binwidth, ndecimal
# print float(binwidth/2.0), mu , sig

xsigup = mu + 2*sig
xsigdown = mu - 2*sig

xfitup = mu + 5*sig
xfitdown = mu - 5*sig

bins = int(round((xfitup-xfitdown)/binwidth))

print xsigup, xsigdown, xfitup, xfitdown, bins
print float(xsigup), float(xsigdown), float(xfitup), float(xfitdown), int(bins) 

def get_chisq(poly="bernstein",order=1,mask=False,saveplot=False,sigshape="dcbg"):
	
	x = ROOT.RooRealVar("x","x",float(xfitdown),float(xfitup))

        x.setRange("R1",float(xfitdown),float(xsigdown))
        x.setRange("R2",float(xsigup),float(xfitup))

        l = ROOT.RooArgList(x)

        data_obs = ROOT.RooDataHist("data_obs", "data_obs", l, data)

        if sigshape == "gaus":

            mean = ROOT.RooRealVar("mean","Mean of Gaussian",mu)
            sigma = ROOT.RooRealVar("sigma","Width of Gaussian",sig)
            signal = ROOT.RooGaussian("signal","signal",x,mean,sigma)

        if sigshape == "bwg":

            mean = ROOT.RooRealVar("mean","Mean of Voigtian",mu)
            gamma = ROOT.RooRealVar("gamma","Gamma of Voigtian",gam)
            sigma = ROOT.RooRealVar("sigma","Sigma of Voigtian",sig)
            signal = ROOT.RooVoigtian("signal","signal",x,mean,gamma,sigma)

        if sigshape == "dcbg":
        
            # mean = ROOT.RooRealVar('mean', 'Mean of DoubleCB', float(mass- (25*binwidth)), float(mass + (25*binwidth)))
            mean = ROOT.RooRealVar('mean', 'Mean of DoubleCB', mu)  
            sigma = ROOT.RooRealVar('sigma', 'Sigma of DoubleCB', sig)
            alpha_1 = ROOT.RooRealVar('alpha_1', 'alpha1 of DoubleCB',  1)
            alpha_2 = ROOT.RooRealVar('alpha_2', 'alpha2 of DoubleCB',  4)
            n_1 = ROOT.RooRealVar('n_1', 'n1 of DoubleCB', 2)
            n_2 = ROOT.RooRealVar('n_2', 'n2 of DoubleCB', 5)

            cbs_1 = ROOT.RooCBShape("CrystallBall_1", "CrystallBall_1", x, mean, sigma, alpha_1, n_1)
            cbs_2 = ROOT.RooCBShape("CrystallBall_2", "CrystallBall_2", x, mean, sigma, alpha_2, n_2)
            mc_frac = ROOT.RooRealVar("mc_frac", "mc_frac", 0.45)

            # signal = ROOT.RooCBShape("signal", "signal", x, mean, sigma, alpha_1, n_1)
            # signal = ROOT.RooAddPdf("signal", "signal", ROOT.RooArgList(cbs_1,cbs_2), ROOT.RooArgList(mc_frac))

            # mean1 = ROOT.RooRealVar("mean1","Mean of Gaussian",float(mass- (25*binwidth)),float(mass + (25*binwidth)))
            mean1 = ROOT.RooRealVar("mean1","Mean of Gaussian", mu)                                                           

            if mass > 2:  
                sigma1 = ROOT.RooRealVar("sigma1","Width of Gaussian",0.35)                                                  
            else:
                sigma1 = ROOT.RooRealVar("sigma1","Width of Gaussian",0.05)    

            gaus = ROOT.RooGaussian("gaus","gaus",x,mean1,sigma1)
            mc_frac1 = ROOT.RooRealVar("mc_frac1", "mc_frac1", 0.5)

            signal = ROOT.RooAddPdf('signal', 'signal', ROOT.RooArgList(cbs_1,cbs_2,gaus), ROOT.RooArgList(mc_frac,mc_frac1))

        nS = ns
        sig_norm = ROOT.RooRealVar("sig_norm","sig_norm",nS,0,10*nS)


	p = [0]*(order+1)
	par = ROOT.RooArgList()

	if poly == "cheb":

                for i in range(order+1):
                        p[i] = ROOT.RooRealVar("p{}".format(i),"p{}".format(i),-1,1)
                        par.add(p[i])
		background = ROOT.RooChebychev("background","background", x, par)


        if poly == "simplepoly":

                for i in range(order+1):
                        p[i] = ROOT.RooRealVar("p{}".format(i),"p{}".format(i),-100000,100000)
                        par.add(p[i])
                background = ROOT.RooPolynomial("background","background", x, par)

        if poly == "bernstein":
                
                for i in range(order+1):
                        p[i] = ROOT.RooRealVar("p{}".format(i),"p{}".format(i),-1,1000000000)
                        par.add(p[i])
                background = ROOT.RooBernstein("background","background", x, par)

        print p

	nB = data.Integral()
        if nB != 0:
            background_norm = ROOT.RooRealVar("background_norm","background_norm",nB,0.9*nB,1.1*nB)
        else:
            background_norm = ROOT.RooRealVar("background_norm","background_norm",nB,0.0,0.000001)

        # background_norm = ROOT.RooRealVar("background_norm","background_norm",nB,0.5*nB,1.5*nB) 
	
	model = ROOT.RooAddPdf("model","model",ROOT.RooArgList(background),ROOT.RooArgList(background_norm))
        # model = ROOT.RooAddPdf("model","model",ROOT.RooArgList(signal,background),ROOT.RooArgList(sig_norm,background_norm))

	# ROOT.RooMsgService.instance().setSilentMode(ROOT.kTRUE)

	if mask:

		result = ROOT.RooFitResult(model.fitTo(data_obs, ROOT.RooFit.Range("R1,R2"), ROOT.RooFit.Save(ROOT.kTRUE), ROOT.RooFit.Minimizer("Minuit2","Migrad")))
		model.fitTo(data_obs,ROOT.RooFit.Range("R1,R2"))
	
	else:

		result = ROOT.RooFitResult(model.fitTo(data_obs, ROOT.RooFit.Range("Full"), ROOT.RooFit.Save(ROOT.kTRUE), ROOT.RooFit.Minimizer("Minuit2","Migrad")))
		model.fitTo(data_obs,ROOT.RooFit.Range("Full"))

	bkg_component = ROOT.RooArgSet(background)  
	xframe = x.frame(ROOT.RooFit.Title("Data Fit"))
        data_obs.plotOn(xframe, ROOT.RooFit.Name("data"))
	model.plotOn(xframe,ROOT.RooFit.LineColor(3),ROOT.RooFit.Name("bkg"), ROOT.RooFit.LineStyle(2), ROOT.RooFit.Range("Full"), ROOT.RooFit.NormRange("Full"))
        model.plotOn(xframe,ROOT.RooFit.LineColor(3),ROOT.RooFit.Name("bkg"), ROOT.RooFit.LineStyle(2), ROOT.RooFit.Range("Full"), ROOT.RooFit.NormRange("Full"))

        chisq = xframe.chiSquare(order)
        nll = result.minNll()

        # model.plotOn(xframe,ROOT.RooFit.VisualizeError(result,1,ROOT.kFALSE), ROOT.RooFit.DrawOption("F"), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange), ROOT.RooFit.Range("Full"), ROOT.RooFit.NormRange("Full"))
        model.plotOn(xframe,ROOT.RooFit.LineColor(3),ROOT.RooFit.Name("bkg"), ROOT.RooFit.LineStyle(2), ROOT.RooFit.Range("Full"), ROOT.RooFit.NormRange("Full"))
        
        xframe.Print("v")

        fitcurve = ROOT.RooCurve(xframe.getCurve("bkg"))
        fitgraph = ROOT.TGraph(fitcurve.GetN())

        for i in range(fitcurve.GetN()):
                fitgraph.SetPoint(i, fitcurve.GetX()[i], fitcurve.GetY()[i])

        rss = 0
        xssq = 0
        print data.GetNbinsX()
        for i in range(data.GetNbinsX()):
                if(data.GetBinCenter(i+1)>float(xfitdown)):
                        #print h2[j].GetBinCenter(i+1)                                                                                                                                                     
                        dataCount  = data.GetBinContent(i+1)
                        fitValue = fitgraph.Eval(data.GetBinCenter(i+1))
                        print data.GetBinCenter(i+1), dataCount, fitValue
                        rs = (dataCount - fitValue)**2
                        if fitValue != 0:
                                xsq = ((dataCount - fitValue)**2)/fitValue
                        else:
                                xsq = 0
                rss += rs
                xssq += xsq
        print rss
        print xssq
        
	myWS = ROOT.RooWorkspace("myWS", "workspace")                                                                                                                                                      
        getattr(myWS,'import')(data_obs)                                                                                                    
        # getattr(myWS,'import')(background)                                                                                                
        getattr(myWS,'import')(signal) 
        getattr(myWS,'import')(model)

        # getattr(myWS,'import')(model,RooCmdArg())
        # getattr(myWS, 'import')(background_norm)
        # getattr(myWS, 'import')(sig_norm)
        # getattr(myWS, 'factory')("nS".format(nS))
        # getattr(myWS, 'factory')("nB".format(nB))        
        # getattr(myWS, 'factory')(ROOT.Sum.()Format("nB",nB))
        

	myWS.writeToFile("simple-shapes-TH1_mass{}_Lxy{}_{}_{}_order{}.root".format(mass, lxybins[j,0],lxybins[j,1],poly,order))                                                                    
        myWS.Print()                                                                                                                                                                                       
        print "RooWorkspace made"                                                                                                                                                                          
        ROOT.gDirectory.Add(myWS)
	
	datacard = open("simple-shapes-TH1_mass{}_Lxy{}_{}_{}_order{}.txt".format(mass, lxybins[j,0],lxybins[j,1],poly,order), "w")                                                                                
        datacard.write("imax 1  number of channels\n")                                                                                                                                                     
        datacard.write("jmax 1  number of backgrounds\n")                                                                                                                                                  
        datacard.write("kmax *  number of nuisance parameters (sources of systematical uncertainties)\n")                                                                                                  
        datacard.write("------------------------------------\n")                                                                                                                                           
        datacard.write("shapes * * simple-shapes-TH1_mass{}_Lxy{}_{}_{}_order{}.root myWS:$PROCESS\n".format(mass, lxybins[j,0],lxybins[j,1],poly,order))                                             
        datacard.write("------------------------------------\n")                                                                                                                                           
        datacard.write("bin bin1\n")                                                                                                                                                                       
        datacard.write("observation -1\n")                                                                                                                                                                 
        datacard.write("------------------------------------\n")                                                                                                                                           
        datacard.write("bin bin1 bin1\n")                                                                                                                                                                  
        datacard.write("process signal background\n")                                                                                                                                                      
        datacard.write("process 0 1\n")                                                                                                                                                                    
        # datacard.write("rate {} {}\n".format(nS, nB))                                                                                                                                                    
        datacard.write("rate {} {}\n".format(nS, 1))

        datacard.write("------------------------------------\n")                                                                                                                                           
        datacard.write("lumi lnN 1.025 1.0\n")                                                                                                                                                             
        datacard.write("bgnorm lnN - 1.05\n")                                                                                                                                                          
        datacard.write("signorm lnN 1.05 -\n")
                                                                           
        datacard.close()                                                                                             


	os.system('combine -M  AsymptoticLimits -m {} --rAbsAcc=0.0001 --rRelAcc=0.001 simple-shapes-TH1_mass{}_Lxy{}_{}_{}_order{}.txt > com.out'.format(mass, mass, lxybins[j,0],lxybins[j,1],poly,order)) 
	os.system('cat com.out')                                                                                                                                                                           
        com_out = open('com.out','r')                                                                                                                                                                       
        for line in com_out:                                                                                                                                                                                
                if line[:15] == 'Observed Limit:':                                                                                                                                                         
                        coml_obs = float(line[19:])                                                                                                                                                         
                elif line[:15] == 'Expected  2.5%:':                                                                                                                                                       
                        coml_2sd = float(line[19:])                                                                                                                                                        
			
                elif line[:15] == 'Expected 16.0%:':                                                                                                                                                       
                        coml_1sd = float(line[19:])                                                                                                                                                         
                elif line[:15] == 'Expected 50.0%:':                                                                                                                                                       
                        coml_exp = float(line[19:])                                                                                                                                                        
			
                elif line[:15] == 'Expected 84.0%:':                                                                                                                                                       
                        coml_1su = float(line[19:])                                                                                                                                                                                                                                                                                                                                                                   
                elif line[:15] == 'Expected 97.5%:':                                                                                                                                                       
                        coml_2su = float(line[19:])     

	os.system("combine -M GoodnessOfFit --algo=saturated -m {} simple-shapes-TH1_mass{}_Lxy{}_{}_{}_order{}.txt".format(mass, mass, lxybins[j,0],lxybins[j,1],poly,order))
	KS_Fs = ROOT.TFile("higgsCombineTest.GoodnessOfFit.mH" + str(mass) + ".root")
	KS_Ts = KS_Fs.Get("limit")
	KS_Vs = []

	for i in range(0, KS_Ts.GetEntries()):
		KS_Ts.GetEntry(i)
		if (KS_Ts.limit < 10000):
			KS_Vs.append(KS_Ts.limit)

        os.system("combine -M GoodnessOfFit --algo=saturated -m {} simple-shapes-TH1_mass{}_Lxy{}_{}_{}_order{}.txt -t {}".format(mass, mass, lxybins[j,0],lxybins[j,1],poly,order,250))
        KS_F = ROOT.TFile("higgsCombineTest.GoodnessOfFit.mH" + str(mass) + ".123456.root")
        KS_T = KS_F.Get("limit")
        KS_V = []

        for i in range(0, KS_T.GetEntries()):
                KS_T.GetEntry(i)
                if (KS_T.limit < 10000):
                        KS_V.append(KS_T.limit)

        # Plot                                                                                                                                                                                               
        minKS = min(min(KS_V), min(KS_Vs))
        maxKS = max(max(KS_V), max(KS_Vs))
        rangeKS = maxKS - minKS
        KS_plot = ROOT.TH1F("KS_plot", "%s;Goodness Of Fit Statistic (Saturated);toys" % ("Goodness of Fit"),50, minKS-(rangeKS/10), maxKS+(rangeKS/10))
        KS_plot.SetStats(0)
        for i in KS_V:
                KS_plot.Fill(i)
        # GoodPlotFormat(KS_plot, "markers", ROOT.kBlack, 20)
        # KS_mk = ROOT.TLine(KS_Vs[0], 0., KS_Vs[0], KS_plot.GetMaximum())
        # KS_plot.Draw()

        integral = KS_plot.Integral(1, KS_plot.FindBin(KS_Vs[0]))


        if saveplot:
		c2 = ROOT.TCanvas("c2","c2")
		pad1 = ROOT.TPad("pad1", "The pad 80% of the height", 0.0, 0.2, 1.0, 1.0, 0)
		pad2 = ROOT.TPad("pad2", "The pad 20% of the height", 0.0, 0.0, 1.0, 0.2, 0)
		c2.cd()

		pad1.Draw()
		pad2.Draw()

		pad1.cd()
		pad1.SetTickx()
		pad1.SetTicky()
		pad1.SetBottomMargin(0.01)
                # AddCMSLumi(ROOT.gPad, 10.1, "pvalue")

		# ROOT.gStyle.SetEndErrorSize(0)
		xframe2 = x.frame(ROOT.RooFit.Title("mass {}GeV, lxy {}cm - {}cm, {}_o({}) fit, ".format(mass,lxybins[j,0], lxybins[j,1],poly,order) + "#chi^{2}/ndf = " + "%2f" %(xssq/(numbins-order)) + ", p = %.3f" % (integral / 250) ))
		data_obs.plotOn(xframe2, ROOT.RooFit.Name("data"))
		model.plotOn(xframe2,ROOT.RooFit.LineColor(2), ROOT.RooFit.LineStyle(2), ROOT.RooFit.Name("bkg"), ROOT.RooFit.Range("Full"), ROOT.RooFit.NormRange("Full"))
		# model.plotOn(xframe2,ROOT.RooFit.VisualizeError(result,1,ROOT.kFALSE), ROOT.RooFit.DrawOption("F"), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange), ROOT.RooFit.Range("Full"), ROOT.RooFit.NormRange("Full"))
                # model.plotOn(xframe2,ROOT.RooFit.VisualizeError(result,1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange), ROOT.RooFit.Range("Full"), ROOT.RooFit.NormRange("Full")) 
		data_obs.plotOn(xframe2, ROOT.RooFit.Name("data"))
		model.plotOn(xframe2,ROOT.RooFit.LineColor(2), ROOT.RooFit.Name("bkg"), ROOT.RooFit.LineStyle(1), ROOT.RooFit.Range("Full"), ROOT.RooFit.NormRange("Full"))

		xframe2.Draw()
		# xframe2.GetYaxis().SetRangeUser(-20,120)                                                                                                                                                 
		xframe2.GetYaxis().SetTitle("Events/ {} GeV".format(binwidth))
		xframe2.GetYaxis().SetTitleSize(0.05)
		xframe2.GetYaxis().SetLabelSize(0.045)
		xframe2.GetYaxis().SetTitleOffset(0.95)

		if mask:
			box = ROOT.TBox(float(xsigdown),xframe2.GetMinimum(),float(xsigup),xframe2.GetMaximum()) 
			box.SetFillColorAlpha(7,0.35)
			box.SetFillStyle(1001)
			box.Draw()

		# leg1 = ROOT.TLegend(0.1,0.6,0.4,0.9)
                leg1 = ROOT.TLegend(0.6,0.0,0.9,0.3)
		leg1.SetLineColor(0)
		leg1.SetFillColor(0)
		leg1.SetFillStyle(0)
		leg1.AddEntry(xframe2.findObject("data"), "Data [{} < lxy < {}]".format(lxybins[j,0], lxybins[j,1]), "pe") 
		leg1.AddEntry(xframe2.findObject("bkg"), "#color[2]{%s Fit}" %(poly + "_o" + "(" + str(order) + ")"), "l")
		# leg1.AddEntry(xframe2.findObject("errorband"), "Fit Error", "f")
		leg1.SetTextFont(42)
		leg1.SetBorderSize(0)
		# leg1.Draw()

		pull = ROOT.RooHist(xframe2.pullHist("data","bkg"))
		pull.SetFillColor(ROOT.kRed)
		pull.SetLineWidth(0)

		xframe3 = x.frame(ROOT.RooFit.Title(" "))
		
		xframe3.addPlotable(pull,"B X")
		xframe3.GetXaxis().SetLabelSize(0.17)
		xframe3.GetYaxis().SetLabelSize(0.15)
		xframe3.GetXaxis().SetTitleSize(0.21)
		xframe3.GetYaxis().SetTitleSize(0.15)
		xframe3.GetXaxis().SetTitleOffset(0.85)
		xframe3.GetYaxis().SetTitleOffset(0.28)
		xframe3.GetXaxis().SetTitle("Dimuon Mass [GeV]")
		xframe3.GetYaxis().SetTitle("#scale[1.3]{#frac{data - fit}{#sigma_{data}}}")
		# xframe3.GetYaxis().SetTitle("Pull")
		xframe3.GetYaxis().SetLabelSize(0.15)

		pad2.cd()
		pad2.SetTickx()
		pad2.SetTicky()
		# pad2.SetGridy()     	
		pad2.SetTopMargin(0.0)
		pad2.SetBottomMargin(0.4)
		xframe3.Draw()

		# c2.BuildLegend()
		c2.Draw()
                # if not os.path.exists("mass{}".format(mass)):
                #         os.makedirs("mass{}".format(mass))
                c2.SaveAs("mass{}_lxy{}_{}_{}_order{}.png".format(mass, lxybins[j,0], lxybins[j,1],poly,order))



        if saveplot:
                GoodPlotFormat(KS_plot, "markers", ROOT.kBlack, 20)
                KS_mk = ROOT.TLine(KS_Vs[0], 0., KS_Vs[0], KS_plot.GetMaximum())
                KS_mk.SetLineColor(ROOT.kRed)
                KS_mk.SetLineWidth(3)

                # Legend                                                                                                                                                                                                
                legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
                legend.SetBorderSize(0)
                legend.SetFillColor(0)
                legend.AddEntry(KS_plot, "Toy Models", "pe")
                legend.AddEntry(KS_mk, "Bg, p = %.3f" % (integral / 250), "l")

                C_KS = ROOT.TCanvas()
                C_KS.cd()
                KS_plot.SetTitle("mass {}GeV, lxy {}cm - {}cm, {}_o({}) fit".format(mass,lxybins[j,0], lxybins[j,1],poly,order))
                KS_plot.Draw("e")
                KS_mk.Draw("same")
                legend.Draw()
                ROOT.gPad.SetTicks(1, 1)
                ROOT.gPad.RedrawAxis()
                # AddCMSLumi(ROOT.gPad, plot_lumi, cmsextra)                                                                                                                        
                
                if not os.path.exists("bias_signalinjection_pvalue"):                                                                                                               
                        os.makedirs("bias_signalinjection_pvalue")  
                C_KS.Print("bias_signalinjection_pvalue/goodnessoffit_mass{}_lxy{}_{}_{}_order{}.png".format(mass, lxybins[j,0], lxybins[j,1],poly,order))

                ################################BiasTest using toys from same polynomial#################################

                INJ = [2.]
                ntoys = 500
                cardname = "simple-shapes-TH1_mass{}_Lxy{}_{}_{}_order{}.txt".format(mass, lxybins[j,0],lxybins[j,1],poly,order)
                name = "analysis"

                for i in INJ:
                        os.system("combine %s -M GenerateOnly -t %d -m %f --saveToys --toysFrequentist --expectSignal %f -n %s%f --bypassFrequentistFit" %(cardname, ntoys, mass, i, name, i))


                        if num_after_point(mass) == 0:

                            os.system("combine -M FitDiagnostics -d %s -m %f --bypassFrequentistFit --skipBOnlyFit -t %d --toysFile higgsCombine%s%f.GenerateOnly.mH%i.123456.root --rMin -5 --rMax %f --saveWorkspace -n %s%f" %(cardname, mass, ntoys, name, i, mass, max(i*5, 5), name, i))


                        elif num_after_point(mass) == 1:

                            os.system("combine -M FitDiagnostics -d %s -m %f --bypassFrequentistFit --skipBOnlyFit -t %d --toysFile higgsCombine%s%f.GenerateOnly.mH%.1f.123456.root --rMin -5 --rMax %f --saveWorkspace -n %s%f" %(cardname, mass, ntoys, name, i, mass, max(i*5, 5), name, i))

                        elif num_after_point(mass) == 2:

                            os.system("combine -M FitDiagnostics -d %s -m %f --bypassFrequentistFit --skipBOnlyFit -t %d --toysFile higgsCombine%s%f.GenerateOnly.mH%.2f.123456.root --rMin -5 --rMax %f --saveWorkspace -n %s%f" %(cardname, mass, ntoys, name, i, mass, max(i*5, 5), name, i))


                        elif num_after_point(mass) == 3:

                            os.system("combine -M FitDiagnostics -d %s -m %f --bypassFrequentistFit --skipBOnlyFit -t %d --toysFile higgsCombine%s%f.GenerateOnly.mH%.3f.123456.root --rMin -5 --rMax %f --saveWorkspace -n %s%f" %(cardname, mass, ntoys, name, i, mass, max(i*5, 5), name, i))



                        F = ROOT.TFile("fitDiagnostics%s%f.root" % (name, i))
                        T = F.Get("tree_fit_sb")
                        
                        H = ROOT.TH1F("Bias Test, injected r="+str(int(i)),
                                      "Bias Test;(r_{measured} - r_{injected})/#sigma_{r};toys", 48, -6., 6.)
                        T.Draw("(r-%f)/rErr>>Bias Test, injected r=%d" %(i, int(i)))

                        # H = ROOT.TH1F("Bias Test, injected r="+str(int(i)),                                                                                                                    
                        #               "Signal Injection Test;r_{measured};toys", 100, -50., 50.)                                                                                               
                        # T.Draw("r>>Bias Test, injected r=%d" %(int(i)))                                                                                                                        
                        
                        G = ROOT.TF1("f"+name+str(i), "gaus(0)", -5., 5.)
                        G.SetParLimits(0, 1, 2500)
                        G.SetParLimits(1, -5, 5)
                        # G.SetParLimits(1, -20, 20)                                                                                                                                             
                        H.Fit(G)
                        ROOT.gStyle.SetOptFit(1111)
                        
                        bias = G.GetParameter(1)
                        biaserr = G.GetParError(1)

                        C_B = ROOT.TCanvas()
                        C_B.cd()
                        H.SetTitle("mass {}GeV, lxy {}cm - {}cm, {}_o({}) fit".format(mass,lxybins[j,0], lxybins[j,1],poly,order))

                        H.SetLineWidth(2)
                        H.Draw("e0")                

                        C_B.SaveAs("bias_signalinjection_pvalue/biastest{}_mass{}_lxy{}_{}_{}_order{}.png".format(i, mass, lxybins[j,0], lxybins[j,1],poly,order))
                ################################################################################

        if saveplot:
                ggphipoly = open("mass{}.csv".format(mass), "a")
                ggphipoly.write(" mass\tlxy bin\tpoly order\tchi2\tndof\tpvalue\tbias\tbias_err\tExpected 2.5%: r < \tExpected 16.0%: r < \tExpected 50.0%: r < \tExpected 84.0%: r < \tExpected 97.5%: r < \tObserved Limit\n")
                ggphipoly.write(" {}\t{} - {}\t{}{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(mass, lxybins[j,0], lxybins[j,1], poly, order, xssq, numbins - order , integral/250, bias, biaserr, coml_2sd, coml_1sd, coml_exp, coml_1su, coml_2su, coml_obs))  



                ggphipoly = open("mass{}_v0.csv".format(mass), "a")
                ggphipoly.write(" mass;lxy bin;poly order;chi2;ndof;pvalue;bias;bias_err;Expected 2.5%: r < ;Expected 16.0%: r < ;Expected 50.0%: r < ;Expected 84.0%: r < ;Expected 97.5%: r < ;Observed Limit\n")
                ggphipoly.write(" {};{} - {};{}{};{};{};{};{};{};{};{};{};{};{};{}\n".format(mass, lxybins[j,0], lxybins[j,1], poly, order, xssq, numbins - order , integral/250,bias, biaserr, coml_2sd, coml_1sd, coml_exp, coml_1su, coml_2su, coml_obs))

        
        # print KS_Vs[0], KS_plot.FindBin(KS_Vs[0]), integral
	if saveplot != 1: 
		os.system('rm simple-shapes-TH1_mass{}_Lxy{}_{}_{}_order{}.txt'.format(mass, lxybins[j,0],lxybins[j,1],poly,order))
		os.system('rm simple-shapes-TH1_mass{}_Lxy{}_{}_{}_order{}.root'.format(mass, lxybins[j,0],lxybins[j,1],poly,order))
		os.system('rm higgsCombineTest.AsymptoticLimits.mH{}.root'.format(mass))
		os.system('rm higgsCombineTest.GoodnessOfFit.mH{}.root'.format(mass))
                os.system('rm higgsCombineTest.GoodnessOfFit.mH{}.123456.root'.format(mass))

        if saveplot == 1:
                os.system('rm higgsCombineTest.AsymptoticLimits.mH{}.root'.format(mass))
                os.system('rm higgsCombineTest.GoodnessOfFit.mH{}.root'.format(mass))
                os.system('rm higgsCombineTest.GoodnessOfFit.mH{}.123456.root'.format(mass))
                os.system('rm higgsCombineanalysis0.000000.FitDiagnostics.mH{}.123456.root'.format(mass))
                os.system('rm higgsCombineanalysis0.000000.GenerateOnly.mH{}.123456.root'.format(mass))
                os.system('rm fitDiagnosticsanalysis0.000000.root')
                os.system('rm higgsCombineanalysis2.000000.FitDiagnostics.mH{}.123456.root'.format(mass))
                os.system('rm higgsCombineanalysis2.000000.GenerateOnly.mH{}.123456.root'.format(mass))
                os.system('rm fitDiagnosticsanalysis2.000000.root')
                os.system('rm higgsCombineanalysis5.000000.FitDiagnostics.mH{}.123456.root'.format(mass))
                os.system('rm higgsCombineanalysis5.000000.GenerateOnly.mH{}.123456.root'.format(mass))
                os.system('rm fitDiagnosticsanalysis5.000000.root')

	return (chisq,nll,coml_exp,KS_Vs[0],integral/250,rss,xssq)



def counting():

        poly = "counting"
        order = "nil"
        nS = ns
        xssq = "nil"
        
        if numsideband == 0:
                nB = 0.0001
        else:
                nB = numsideband 

        print numdata, numsideband, numsideband*0.67


        datacard = open("simple-shapes-TH1_mass{}_Lxy{}_{}_{}_order{}.txt".format(mass, lxybins[j,0],lxybins[j,1],poly,order), "w")                                    
        datacard.write("imax 1  number of channels\n")                                                                                                                              
        datacard.write("jmax 1  number of backgrounds\n")                                                                                                                           
        datacard.write("kmax *  number of nuisance parameters (sources of systematical uncertainties)\n")                                                                           
        datacard.write("------------------------------------\n")                                                                                                                    
        datacard.write("bin bin1\n")                                                                                                                                                
        datacard.write("observation {}\n".format(numdata-numsideband))                                                                                                                                         
        datacard.write("------------------------------------\n")                                                                                                                   
        datacard.write("bin bin1 bin1\n")                                                                                                                                          
        datacard.write("process signal background\n")                                                                                                                              
        datacard.write("process 0 1\n")                                                                                                                                            
        # datacard.write("rate {} {}\n".format(nS, nB))                                                                                                                             
        datacard.write("rate {} {}\n".format(nS, nB*0.67))
        
        datacard.write("------------------------------------\n")                                                                                                                    
        datacard.write("lumi lnN 1.025 1.0\n")                                                                                                                                      
        datacard.write("bgnorm lnN - 1.5\n")                                                                                                                           
        datacard.write("signorm lnN 1.05 -\n") 
        datacard.write("sb%i gmN %i - 0.67\n" %(j+1,numsideband))
        
        datacard.close()                                                                                             



        os.system('combine -M  AsymptoticLimits -m {} --rAbsAcc=0.0001 --rRelAcc=0.001 simple-shapes-TH1_mass{}_Lxy{}_{}_{}_order{}.txt > com.out'.format(mass, mass, lxybins[j,0],lxybins[j,1],poly,order)) 
	os.system('cat com.out')                                                                                                                                                                           
        com_out = open('com.out','r')                                                                                                                                                                       
        for line in com_out:                                                                                                                                                                                
                if line[:15] == 'Observed Limit:':                                                                                                                                                         
                        coml_obs = float(line[19:])                                                                                                                                                         
                elif line[:15] == 'Expected  2.5%:':                                                                                                                                                       
                        coml_2sd = float(line[19:])                                                                                                                                                        
			
                elif line[:15] == 'Expected 16.0%:':                                                                                                                                                       
                        coml_1sd = float(line[19:])                                                                                                                                                         
                elif line[:15] == 'Expected 50.0%:':                                                                                                                                                       
                        coml_exp = float(line[19:])                                                                                                                                                        
			
                elif line[:15] == 'Expected 84.0%:':                                                                                                                                                       
                        coml_1su = float(line[19:])                                                                                                                                                                                                                                                                                                                                                                   
                elif line[:15] == 'Expected 97.5%:':                                                                                                                                                       
                        coml_2su = float(line[19:])     

	os.system("combine -M GoodnessOfFit --algo=saturated -m {} simple-shapes-TH1_mass{}_Lxy{}_{}_{}_order{}.txt".format(mass, mass, lxybins[j,0],lxybins[j,1],poly,order))
	KS_Fs = ROOT.TFile("higgsCombineTest.GoodnessOfFit.mH" + str(mass) + ".root")
	KS_Ts = KS_Fs.Get("limit")
	KS_Vs = []

	for i in range(0, KS_Ts.GetEntries()):
		KS_Ts.GetEntry(i)
		if (KS_Ts.limit < 10000):
			KS_Vs.append(KS_Ts.limit)

        os.system("combine -M GoodnessOfFit --algo=saturated -m {} simple-shapes-TH1_mass{}_Lxy{}_{}_{}_order{}.txt -t {}".format(mass, mass, lxybins[j,0],lxybins[j,1],poly,order,250))
        KS_F = ROOT.TFile("higgsCombineTest.GoodnessOfFit.mH" + str(mass) + ".123456.root")
        KS_T = KS_F.Get("limit")
        KS_V = []

        for i in range(0, KS_T.GetEntries()):
                KS_T.GetEntry(i)
                if (KS_T.limit < 10000):
                        KS_V.append(KS_T.limit)

        # Plot                                                                                                                                                                                               
        minKS = min(min(KS_V), min(KS_Vs))
        maxKS = max(max(KS_V), max(KS_Vs))
        rangeKS = maxKS - minKS
        KS_plot = ROOT.TH1F("KS_plot", "%s;Goodness Of Fit Statistic (Saturated);toys" % ("Goodness of Fit"),50, minKS-(rangeKS/10), maxKS+(rangeKS/10))
        KS_plot.SetStats(0)
        for i in KS_V:
                KS_plot.Fill(i)
        # GoodPlotFormat(KS_plot, "markers", ROOT.kBlack, 20)
        # KS_mk = ROOT.TLine(KS_Vs[0], 0., KS_Vs[0], KS_plot.GetMaximum())
        # KS_plot.Draw()

        integral = KS_plot.Integral(1, KS_plot.FindBin(KS_Vs[0]))

        # if saveplot:
        GoodPlotFormat(KS_plot, "markers", ROOT.kBlack, 20)
        KS_mk = ROOT.TLine(KS_Vs[0], 0., KS_Vs[0], KS_plot.GetMaximum())
        KS_mk.SetLineColor(ROOT.kRed)
        KS_mk.SetLineWidth(3)
        
        # Legend                                                                                                                                                                                                
        legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
        legend.SetBorderSize(0)
        legend.SetFillColor(0)
        legend.AddEntry(KS_plot, "Toy Models", "pe")
        legend.AddEntry(KS_mk, "Bg, p = %.3f" % (integral / 250), "l")

        C_KS = ROOT.TCanvas()
        C_KS.cd()
        KS_plot.SetTitle("mass {}GeV, lxy {}cm - {}cm, {}_o({}) fit".format(mass,lxybins[j,0], lxybins[j,1],poly,order))
        KS_plot.Draw("e")
        KS_mk.Draw("same")
        legend.Draw()
        ROOT.gPad.SetTicks(1, 1)
        ROOT.gPad.RedrawAxis()
        # AddCMSLumi(ROOT.gPad, plot_lumi, cmsextra)                                                                                                                        
                
        if not os.path.exists("bias_signalinjection_pvalue"):                                                                                                               
                os.makedirs("bias_signalinjection_pvalue")  
        C_KS.Print("bias_signalinjection_pvalue/goodnessoffit_mass{}_lxy{}_{}_{}_order{}.png".format(mass, lxybins[j,0], lxybins[j,1],poly,order))

        INJ = [0.,2.,5.]
        ntoys = 500
        cardname = "simple-shapes-TH1_mass{}_Lxy{}_{}_{}_order{}.txt".format(mass, lxybins[j,0],lxybins[j,1],poly,order)
        name = "analysis"

        for i in INJ:
                os.system("combine %s -M GenerateOnly -t %d -m %f --saveToys --toysFrequentist --expectSignal %f -n %s%f --bypassFrequentistFit" %(cardname, ntoys, mass, i, name, i))

                if num_after_point(mass) == 0:

                    os.system("combine -M FitDiagnostics -d %s -m %f --bypassFrequentistFit --skipBOnlyFit -t %d --toysFile higgsCombine%s%f.GenerateOnly.mH%i.123456.root --rMin -5 --rMax %f --saveWorkspace -n %s%f --forceRecreateNLL" %(cardname, mass, ntoys, name, i, mass, max(i*5, 5), name, i))


                elif num_after_point(mass) == 1:

                    os.system("combine -M FitDiagnostics -d %s -m %f --bypassFrequentistFit --skipBOnlyFit -t %d --toysFile higgsCombine%s%f.GenerateOnly.mH%.1f.123456.root --rMin -5 --rMax %f --saveWorkspace -n %s%f --forceRecreateNLL" %(cardname, mass, ntoys, name, i, mass, max(i*5, 5), name, i))

                elif num_after_point(mass) == 2:

                    os.system("combine -M FitDiagnostics -d %s -m %f --bypassFrequentistFit --skipBOnlyFit -t %d --toysFile higgsCombine%s%f.GenerateOnly.mH%.2f.123456.root --rMin -5 --rMax %f --saveWorkspace -n %s%f --forceRecreateNLL" %(cardname, mass, ntoys, name, i, mass, max(i*5, 5), name, i))

                elif num_after_point(mass) == 3:

                    os.system("combine -M FitDiagnostics -d %s -m %f --bypassFrequentistFit --skipBOnlyFit -t %d --toysFile higgsCombine%s%f.GenerateOnly.mH%.3f.123456.root --rMin -5 --rMax %f --saveWorkspace -n %s%f --forceRecreateNLL" %(cardname, mass, ntoys, name, i, mass, max(i*5, 5), name, i))


                F = ROOT.TFile("fitDiagnostics%s%f.root" % (name, i))
                T = F.Get("tree_fit_sb")
                
                H = ROOT.TH1F("Bias Test, injected r="+str(int(i)),
                              "Bias Test;(r_{measured} - r_{injected})/#sigma_{r};toys", 48, -6., 6.)
                T.Draw("(r-%f)/rErr>>Bias Test, injected r=%d" %(i, int(i)))

                # H = ROOT.TH1F("Bias Test, injected r="+str(int(i)),                                                                                                                    
                #               "Signal Injection Test;r_{measured};toys", 100, -50., 50.)                                                                                               
                # T.Draw("r>>Bias Test, injected r=%d" %(int(i)))                                                                                                                        
                        
                G = ROOT.TF1("f"+name+str(i), "gaus(0)", -5., 5.)
                G.SetParLimits(0, 1, 2500)
                G.SetParLimits(1, -5, 5)
                # G.SetParLimits(1, -20, 20)                                                                                                                                             
                H.Fit(G)
                ROOT.gStyle.SetOptFit(1111)
                
                bias = G.GetParameter(1)
                biaserr = G.GetParError(1)

                C_B = ROOT.TCanvas()
                C_B.cd()
                H.SetTitle("mass {}GeV, lxy {}cm - {}cm, {}_o({}) fit".format(mass,lxybins[j,0], lxybins[j,1],poly,order))
                
                H.SetLineWidth(2)
                H.Draw("e0")                
                
                C_B.SaveAs("bias_signalinjection_pvalue/biastest{}_mass{}_lxy{}_{}_{}_order{}.png".format(i, mass, lxybins[j,0], lxybins[j,1],poly,order))
                

        ggphipoly = open("mass{}.csv".format(mass), "a")
        ggphipoly.write(" mass\tlxy bin\tpoly order\tchi2\tndof\tpvalue\tbias\tbias_err\tExpected 2.5%: r < \tExpected 16.0%: r < \tExpected 50.0%: r < \tExpected 84.0%: r < \tExpected 97.5%: r < \tObserved Limit\n")
        ggphipoly.write(" {}\t{} - {}\t{}{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(mass, lxybins[j,0], lxybins[j,1], poly, order, xssq, order , integral/250, bias, biaserr, coml_2sd, coml_1sd, coml_exp, coml_1su, coml_2su, coml_obs))  


        ggphipoly = open("mass{}_v0.csv".format(mass), "a")
        ggphipoly.write(" mass;lxy bin;poly order;chi2;ndof;pvalue;bias;bias_err;Expected 2.5%: r < ;Expected 16.0%: r < ;Expected 50.0%: r < ;Expected 84.0%: r < ;Expected 97.5%: r < ;Observed Limit\n")
        ggphipoly.write(" {};{} - {};{}{};{};{};{};{};{};{};{};{};{};{};{}\n".format(mass, lxybins[j,0], lxybins[j,1], poly, order, xssq, order , integral/250,bias, biaserr, coml_2sd, coml_1sd, coml_exp, coml_1su, coml_2su, coml_obs))


        os.system('rm higgsCombineTest.AsymptoticLimits.mH{}.root'.format(mass))
        os.system('rm higgsCombineTest.GoodnessOfFit.mH{}.root'.format(mass))
        os.system('rm higgsCombineTest.GoodnessOfFit.mH{}.123456.root'.format(mass))
        os.system('rm higgsCombineanalysis0.000000.FitDiagnostics.mH{}.123456.root'.format(mass))
        os.system('rm higgsCombineanalysis0.000000.GenerateOnly.mH{}.123456.root'.format(mass))
        os.system('rm fitDiagnosticsanalysis0.000000.root')
        os.system('rm higgsCombineanalysis2.000000.FitDiagnostics.mH{}.123456.root'.format(mass))
        os.system('rm higgsCombineanalysis2.000000.GenerateOnly.mH{}.123456.root'.format(mass))
        os.system('rm fitDiagnosticsanalysis2.000000.root')
        os.system('rm higgsCombineanalysis5.000000.FitDiagnostics.mH{}.123456.root'.format(mass))
        os.system('rm higgsCombineanalysis5.000000.GenerateOnly.mH{}.123456.root'.format(mass))
        os.system('rm fitDiagnosticsanalysis5.000000.root')


#################################################Main###########################################################


h2 = []
h1 = []

for j in range(len(lxybins)):

        print "Looking at lxy bin----------",lxybins[j,0], "-", lxybins[j,1], "----------------"   
        
        ns = 100

        h2.append(ROOT.TH1F("h2[{}]".format(j),"h2[{}]".format(j), int(bins), float(xfitdown), float(xfitup)))

        tree_mudata.Draw('mass>>h2[{}]'.format(j),"lxy > {} && lxy < {} && mass > {} && mass < {}".format(lxybins[j,0], lxybins[j,1], xfitdown, xfitup),'')

        print float(xfitdown), float(xfitup)

        x_unmasked = []
        y_unmasked = []
        print h2[j].GetNbinsX()
        for i in range(h2[j].GetNbinsX()):
                if(h2[j].GetBinCenter(i+1)>float(xfitdown)): 
                        #print h2[j].GetBinCenter(i+1)
                        x_unmasked.append(round(h2[j].GetBinCenter(i+1),ndecimal))
                        y_unmasked.append(h2[j].GetBinContent(i+1))

        # print "x_unmasked", x_unmasked
        # print "y_unmasked", y_unmasked
        # print "y_unmasked_error", np.sqrt(y_unmasked)

        data = ROOT.TH1F("data","Histogram of data_obs__x", int(bins), float(xfitdown), float(xfitup))
        for i in range(h2[j].GetNbinsX()):
                # print i, x_unmasked[i],y_unmasked[i]
                data.SetBinContent(i+1,y_unmasked[i])

        h1.append(ROOT.TH1F("h1[{}]".format(j),"h1[{}]".format(j), int(bins), float(xfitdown), float(xfitup)))

        tree_mudata.Draw('mass>>h1[{}]'.format(j),"lxy > {} && lxy < {} && mass > {} && mass < {} && (mass < {} || mass > {})".format(lxybins[j,0], lxybins[j,1], xfitdown, xfitup, xsigdown, xsigup),'')

        x_masked = []
        y_masked = []
        print h1[j].GetNbinsX()
        for i in range(h1[j].GetNbinsX()):
                # if(h1[j].GetBinCenter(i+1)>float(xfitdown) and h1[j].GetBinContent(i+1)>0):
                #         x_masked.append(round(h1[j].GetBinCenter(i+1),3))
                #         y_masked.append(h1[j].GetBinContent(i+1))

                if(h1[j].GetBinCenter(i+1)>float(xfitdown) and h1[j].GetBinCenter(i+1)<float(xsigdown)):
                        x_masked.append(round(h1[j].GetBinCenter(i+1),ndecimal))
                        y_masked.append(h1[j].GetBinContent(i+1))

                # if(h1[j].GetBinCenter(i+1)>=float(xsigdown) and h1[j].GetBinCenter(i+1)<=float(xsigup) and h1[j].GetBinContent(i+1)>0):
                #         x_masked.append(round(h1[j].GetBinCenter(i+1),3))
                #         y_masked.append(h1[j].GetBinContent(i+1))
 
                if(h1[j].GetBinCenter(i+1)>float(xsigup) and h1[j].GetBinCenter(i+1)<float(xfitup)):
                        x_masked.append(round(h1[j].GetBinCenter(i+1),ndecimal))
                        y_masked.append(h1[j].GetBinContent(i+1))

        # print "x_masked", x_masked
        # print "y_masked", y_masked
        # print "y_masked_error", np.sqrt(y_masked)

        x_sigdata = [] 
        y_sigdata = []
        for i in range(h2[j].GetNbinsX()):
                if(h2[j].GetBinCenter(i+1)>float(xsigdown) and h2[j].GetBinCenter(i+1)<float(xsigup)):
                        x_sigdata.append(round(h2[j].GetBinCenter(i+1),ndecimal))
                        y_sigdata.append(h2[j].GetBinContent(i+1))


        numbins = len(x_unmasked) 
        numdata = sum(y_unmasked)
        numsideband = sum(y_masked)
        # if numdata == 0:
        #         numdata = 0.00001


###########################################F-Test,Fit,Count###########################################################
        
        print "number of events", numdata
        print "number of sideband events", numsideband
        print "number of bins", numbins

        ##################

        # counting()

        ##################
        '''
        polytype = "bernstein"

        import scipy.stats                                                                                                                  
        from array import array     
        numbins = len(x_unmasked)                                                                                                                                                     
        rss = 0
        for o in range(10):
            rss0 = rss
            residuals = get_chisq(poly=polytype,order=o+1,mask=False,saveplot=False,sigshape="dcbg") 
            rss = residuals[5] 
            ndf = numbins - (o+1)
            fvalue = (rss0 - rss)/(rss/ndf)                                                                                                                                     
            fcrit = scipy.stats.f.ppf(q=1-0.05, dfn=1, dfd=ndf) 
            
            if o+1 > 1 and fvalue < fcrit:                                                                                                                                     
                break                                                                                                                                                      
        bestorder = o                                                                                                                                                       
        print bestorder        
                
        get_chisq(poly=polytype,order=bestorder,mask=False,saveplot=True,sigshape="dcbg")

        '''
        # '''

        polytype = "bernstein"

        if numsideband < 20:

                counting()


        else:

                import scipy.stats                                                                                                                                                          
                from array import array     
        
                numbins = len(x_unmasked)                                                                                                                                                     
                rss = 0
                for o in range(10):
                        rss0 = rss
                        residuals = get_chisq(poly=polytype,order=o+1,mask=False,saveplot=False) 
                        rss = residuals[5] 
                        ndf = numbins - (o+1)
                        fvalue = (rss0 - rss)/(rss/ndf)                                                                                                                                     
                        fcrit = scipy.stats.f.ppf(q=1-0.05, dfn=1, dfd=ndf) 
                
                        if o+1 > 1 and fvalue < fcrit:                                                                                                                                      
                                break                                                                                                                                                      
                bestorder = o                                                                                                                                                       
                print bestorder        
                
                get_chisq(poly=polytype,order=bestorder,mask=False,saveplot=True)


        # '''

os.chdir("./..")
