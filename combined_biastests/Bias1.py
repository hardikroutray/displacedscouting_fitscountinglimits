import PyFunctions
from PyFunctions import *
import ROOT
import math
import glob
import sys
import csv
import pandas as pd

from array import array
import re
import json
import types
import os
import numpy as np

def num_after_point(x):
    s = str(x)
    if not '.' in s:
        return 0
    return len(s) - s.index('.') - 1


masseslist = [0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7,0.725,0.75,0.775,0.8,0.825,0.85,0.875,0.9,0.925,0.95,1.25,1.5,1.75,2,2.24,2.4,2.5,2.6,2.76,3,3.25,3.5,3.75,4,4.24,4.52,4.76,5,5.5,6,6.6,7,7.49,8,8.4,9,9.5,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

# masseslist = [2]

masses = []
for i in range(len(masseslist)):
    if (masseslist[i] > 0.41 and masseslist[i] < 0.515) or (masseslist[i] > 0.495 and masseslist[i] < 0.61) or (masseslist[i] > 0.695 and masseslist[i] < 0.88) or (masseslist[i] > 0.915 and masseslist[i] < 1.13) or (masseslist[i] > 2.81 and masseslist[i] < 4.09) or (masseslist[i] > 8.59 and masseslist[i] < 11.27):
        continue
    masses.append(masseslist[i])


print len(masses)
# print masses                                                                                 

                                               
mass = masses[int(sys.argv[1])]
ctaus = [1,10,100]
# ctaus = [1,100]

print "Running on mass", mass

os.chdir("mass_{}".format(mass))

for j in range(len(ctaus)):

        print "Looking at ctau----------",ctaus[j], "----------------"

        # os.chdir("./mass_{}".format(mass))
        for file in glob.glob("datacard*allbinsbias.txt"):
                if "ctau{}_".format(ctaus[j]) in file:
                        print(file)
                        cardname = file
        outname = (cardname.split("datacard_")[1]).split(".txt")[0]
        print outname

        bias0= []
        bias0err = []
        bias2 = []
        bias2err = []
        bias5 = []
        bias5err = []

        bias0.append(mass)
        bias0err.append(mass)
        bias2.append(mass)
        bias2err.append(mass)
        bias5.append(mass)
        bias5err.append(mass)


        bias0.append(ctaus[j])
        bias0err.append(ctaus[j])
        bias2.append(ctaus[j])
        bias2err.append(ctaus[j])
        bias5.append(ctaus[j])
        bias5err.append(ctaus[j])

        # Signal Injection
        ntoys = 50
        # name = "analysis"
        INJ = [0.,2.,5.]


        for i in INJ:

            if i == 0:
                rmin = 0
                rmax = 20
            elif i == 2:
                rmin = 0
                rmax = 20
            elif i == 5:
                rmin = 0
                rmax = 20

            # os.system("combine %s -M GenerateOnly -t %d -m %f --saveToys --toysFrequentist --expectSignal %f -n %s%f --bypassFrequentistFit" %(cardname, ntoys, mass, i, name, i))

            nbatches = 5
            for batch in range(1,nbatches+1):

                print "running on batch", batch

                name = "0{}_{}_analysis".format(batch,ctaus[j])

                toycmd = "combine %s -M GenerateOnly -t %d -m %f --saveToys --toysFrequentist --expectSignal %f -n %s%f --bypassFrequentistFit --setParameters   pdf_index223=0,pdf_index312=0,pdf_index111=0,pdf_index112=0,pdf_index113=0,pdf_index121=0,pdf_index122=0,pdf_index123=0,pdf_index211=0,pdf_index212=0,pdf_index213=0,pdf_index221=0,pdf_index222=0,pdf_index311=0,pdf_index313=0,pdf_index321=0,pdf_index322=0,pdf_index323=0,pdf_index411=0,pdf_index412=0,pdf_index413=0,pdf_index421=0,pdf_index422=0,pdf_index423=0,pdf_index511=0,pdf_index512=0,pdf_index513=0,pdf_index521=0,pdf_index522=0,pdf_index523=0,pdf_index611=0,pdf_index612=0,pdf_index613=0,pdf_index621=0,pdf_index622=0,pdf_index623=0   --freezeParameters   pdf_index223,pdf_index312,pdf_index111,pdf_index112,pdf_index113,pdf_index121,pdf_index122,pdf_index123,pdf_index211,pdf_index212,pdf_index213,pdf_index221,pdf_index222,pdf_index311,pdf_index313,pdf_index321,pdf_index322,pdf_index323,pdf_index411,pdf_index412,pdf_index413,pdf_index421,pdf_index422,pdf_index423,pdf_index511,pdf_index512,pdf_index513,pdf_index521,pdf_index522,pdf_index523,pdf_index611,pdf_index612,pdf_index613,pdf_index621,pdf_index622,pdf_index623 -s %i" %(cardname, ntoys, mass, i, name, i, batch)

                os.system(toycmd)

                fitcmd0 = "combine -M FitDiagnostics -d %s -m %f -t %d --toysFile higgsCombine%s%f.GenerateOnly.mH%i.%i.root --rMin %f --rMax %f --saveWorkspace -n %s%f --forceRecreateNLL --keepFailures --saveNormalizations   --savePredictionsPerToy  --saveWithUncertainties --cminDefaultMinimizerStrategy=0" %(cardname, mass, ntoys, name, i, mass, batch, rmin, rmax, name, i)

                fitcmd1 = "combine -M FitDiagnostics -d %s -m %f -t %d --toysFile higgsCombine%s%f.GenerateOnly.mH%.1f.%i.root --rMin %f --rMax %f --saveWorkspace -n %s%f --forceRecreateNLL --keepFailures --saveNormalizations   --savePredictionsPerToy  --saveWithUncertainties --cminDefaultMinimizerStrategy=0" %(cardname, mass, ntoys, name, i, mass, batch, rmin, rmax, name, i)

                fitcmd2 = "combine -M FitDiagnostics -d %s -m %f -t %d --toysFile higgsCombine%s%f.GenerateOnly.mH%.2f.%i.root --rMin %f --rMax %f --saveWorkspace -n %s%f --forceRecreateNLL --keepFailures --saveNormalizations   --savePredictionsPerToy  --saveWithUncertainties --cminDefaultMinimizerStrategy=0" %(cardname, mass, ntoys, name, i, mass, batch, rmin, rmax, name, i)

                fitcmd3 = "combine -M FitDiagnostics -d %s -m %f -t %d --toysFile higgsCombine%s%f.GenerateOnly.mH%.3f.%i.root --rMin %f --rMax %f --saveWorkspace -n %s%f --forceRecreateNLL --keepFailures --saveNormalizations   --savePredictionsPerToy  --saveWithUncertainties --cminDefaultMinimizerStrategy=0" %(cardname, mass, ntoys, name, i, mass, batch, rmin, rmax, name, i)


                if num_after_point(mass) == 0:
                    os.system(fitcmd0)

                elif num_after_point(mass) == 1:
                    os.system(fitcmd1)

                elif num_after_point(mass) == 2:
                    os.system(fitcmd2)

                elif num_after_point(mass) == 3:
                    os.system(fitcmd3)

            
            # Plot
            # F = ROOT.TFile("fitDiagnostics%s%f.root" % (name, i))
            # T = F.Get("tree_fit_sb")

            T = ROOT.TChain("tree_fit_sb")
            T.Add("fitDiagnostics0*_%i_analysis%f.root" %(ctaus[j],i))

            H = ROOT.TH1F("Bias Test, injected r="+str(int(i)),
                          "Bias Test;(r_{measured} - r_{injected})/#sigma_{r};toys", 100, -5., 5.)
            # T.Draw("(r-%f)/rErr>>Bias Test, injected r=%d" %(i, int(i)))

            T.Draw("(r-%f)/((r<%f)*rHiErr + (r>%f)*rLoErr)>>Bias Test, injected r=%d" %(i,i,i,int(i)))

            # H = ROOT.TH1F("Bias Test, injected r="+str(int(i)),
            #               "Signal Injection Test;r_{measured};toys", 100, -50., 50.)
            # T.Draw("r>>Bias Test, injected r=%d" %(int(i)))

            G = ROOT.TF1("f"+name+str(i), "gaus(0)", -5., 5.)
            G.SetParLimits(0, 1, 2500)
            G.SetParLimits(1, -5., 5.)
            # G.SetParLimits(1, -20, 20)
            H.Fit(G,"L")
            ROOT.gStyle.SetOptFit(1111)
            C_B = ROOT.TCanvas()
            C_B.cd()
    
            H.SetLineWidth(2)
            H.Draw("e0")

            H.SetTitle("mass {}GeV, ctau {}mm, envelope on bestbern".format(mass,ctaus[j]))

            if not os.path.exists("bias_signalinjection_pvalue"):    
                os.makedirs("bias_signalinjection_pvalue")

            C_B.SaveAs("bias_signalinjection_pvalue/biastest{}_{}.png".format(i,outname))
            #C_B.Close()
            #C_B.Print(cardname+".png")
            # os.system("rm *.out")
            # os.system("rm *.root")

            if i == 0:

                bias0.append(H.GetMean())
                bias0err.append(H.GetStdDev())

            elif i == 2:

                bias2.append(H.GetMean())
                bias2err.append(H.GetStdDev())

            elif i == 5:

                bias5.append(H.GetMean())
                bias5err.append(H.GetStdDev())




            # os.system('rm higgsCombineanalysis2.000000.FitDiagnostics.mH{}.123456.root'.format(mass))
            # os.system('rm higgsCombineanalysis2.000000.GenerateOnly.mH{}.123456.root'.format(mass))
            # os.system('rm fitDiagnosticsanalysis2.000000.root')


            # os.system('rm higgsCombineanalysis0.000000.FitDiagnostics.mH{}.123456.root'.format(mass))
            # os.system('rm higgsCombineanalysis0.000000.GenerateOnly.mH{}.123456.root'.format(mass))
            # os.system('rm fitDiagnosticsanalysis0.000000.root')


        df_bias0 = pd.DataFrame([bias0],columns=['mass', 'ctau', 'bern'])
        df_bias0err = pd.DataFrame([bias0err],columns=['mass', 'ctau', 'bern'])
        df_bias2 = pd.DataFrame([bias2],columns=['mass', 'ctau', 'bern'])
        df_bias2err = pd.DataFrame([bias2err],columns=['mass', 'ctau', 'bern'])
        df_bias5 = pd.DataFrame([bias5],columns=['mass', 'ctau', 'bern'])
        df_bias5err = pd.DataFrame([bias5err],columns=['mass', 'ctau', 'bern'])

        if not os.path.exists("biascsvs"):
            os.makedirs("biascsvs")

        # if not os.path.exists("rootfiles"):
        #     os.makedirs("rootfiles")

        df_bias0.to_csv('biascsvs/bias0_mass{}_ctau{}_v0.csv'.format(mass,ctaus[j]),index=False)
        df_bias0err.to_csv('biascsvs/bias0err_mass{}_ctau{}_v0.csv'.format(mass,ctaus[j]),index=False)
        df_bias2.to_csv('biascsvs/bias2_mass{}_ctau{}_v0.csv'.format(mass,ctaus[j]),index=False)
        df_bias2err.to_csv('biascsvs/bias2err_mass{}_ctau{}_v0.csv'.format(mass,ctaus[j]),index=False)
        df_bias5.to_csv('biascsvs/bias5_mass{}_ctau{}_v0.csv'.format(mass,ctaus[j]),index=False)
        df_bias5err.to_csv('biascsvs/bias5err_mass{}_ctau{}_v0.csv'.format(mass,ctaus[j]),index=False)


os.chdir("./..")


