import ROOT
import pandas as pd
import csv
import os
import sys
from array import array

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import glob
from matplotlib.colors import LogNorm

from ipywidgets import *
import numpy, scipy, scipy.optimize
import matplotlib
# from mpl_toolkits.mplot3d import  Axes3D                                                      
from matplotlib import cm # to colormap 3D surfaces from blue to red                           

import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

graphWidth = 800 # units are pixels                                                            
graphHeight = 600 # units are pixels                                                           

# 3D contour plot lines                                                                        
numberOfContourLines = 16

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("index_number", help="list index to run on",type=int)
# args = parser.parse_args()

hzdallcsv = pd.read_csv('totalacceptance_hzd.csv')
hzdbincsv = pd.read_csv('binbybinacceptance_hzd.csv')

x = []
y = []
zexp50 = []
accbr = []
acc = []

masseslist = [0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7,0.725,0.75,0.775,0.8,0.825,0.85,0.875,0.9,0.925,0.95,1.25,1.5,1.75,2,2.24,2.4,2.5,2.6,2.76,3,3.25,3.5,3.75,4,4.24,4.52,4.76,5,5.5,6,6.6,7,7.49,8,8.4,9,9.5,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

# masseslist = [25]

print len(masseslist)

masses1 = []
for i in range(len(masseslist)):
    if (masseslist[i] < 0.5) or (masseslist[i] > 0.41 and masseslist[i] < 0.515) or (masseslist[i] > 0.495 and masseslist[i] < 0.61) or (masseslist[i] > 0.695 and masseslist[i] < 0.88) or (masseslist[i] > 0.915 and masseslist[i] < 1.13) or (masseslist[i] > 2.81 and masseslist[i] < 4.09) or (masseslist[i] > 8.59 and masseslist[i] < 11.27):
        continue
    masses1.append(masseslist[i])

print len(masses1)

masses = [masses1[int(sys.argv[1])]]

# ctaus = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1,2,3,4,5,6,7,8,9,10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
ctaus = [1, 10, 100]
# ctaus = [100]

# card = "simple-shapes-TH1_mass2_Lxy1.0_2.4_pt0_25_1isomu_bernstein_order2.txt"

def getbin(inputcard = "card"):
    lxybinstr = ["Lxy0.0_0.2", "Lxy0.2_1.0", "Lxy1.0_2.4", "Lxy2.4_3.1", "Lxy3.1_7.0", "Lxy7.0_11.0"]
    ptbinstr = ["pt0_25", "pt25_Inf"]
    isobinstr = ["2isomu", "1isomu", "0isomu"]

    for i in range(len(lxybinstr)):
        if lxybinstr[i] in inputcard:
            str1 = "lxybin{}_".format(i+1)
    for j in range(len(ptbinstr)):
        if ptbinstr[j] in inputcard:
            str2 = "ptbin{}_".format(j+1)
    for k in range(len(isobinstr)):
        if isobinstr[k] in inputcard:
            str3 = "isobin{}".format(k+1)

    return str1 + str2 + str3

# print getbin(inputcard=card)

def acceptance(mass = 2, ctau = 1, sample="hzd",whichbin="lxybin1_ptbin1_isobin1"):

    if sample == "hzd":
        if whichbin != "allbins":
            x_data = hzdbincsv['mass'].tolist()
            y_data = hzdbincsv['ctau'].tolist()
            z_data = hzdbincsv['acc_'+whichbin].tolist()
        elif whichbin == "allbins":
            x_data = hzdallcsv['mass'].tolist()
            y_data = hzdallcsv['ctau'].tolist()
            z_data = hzdallcsv['acc_'+whichbin].tolist()

    if sample == "bphi":
        if whichbin != "allbins":
            x_data = bphibincsv['mass'].tolist()
            y_data = bphibincsv['ctau'].tolist()
            z_data = bphibincsv['acc_'+whichbin].tolist()
        elif whichbin == "allbins":
            x_data = bphiallcsv['mass'].tolist()
            y_data = bphiallcsv['ctau'].tolist()
            z_data = bphiallcsv['acc_'+whichbin].tolist()

    acc = Rbf(x_data, y_data, z_data, epsilon=0.03)

    return acc(mass,ctau)

# print "the total acceptance", float(acceptance(2,1,sample="hzd",whichbin="allbins"))
# print "the bin acceptance", float(acceptance(2,1,sample="hzd",whichbin=getbin(inputcard=card)))

def get_systematics(mass = 2, ctau = 1, sample="hzd"):

    if sample == "hzd":
        x_data = hzdallcsv['mass'].tolist()
        y_data = hzdallcsv['ctau'].tolist()
        z_data_trigsfunc = hzdallcsv['trigsf_unc'].tolist()
        z_data_oneminuseff = hzdallcsv['oneminuseff'].tolist()
        # trigsf_err = hzdbincsv.loc[(tree_mucsv['mass'] == masses[j]) & (tree_mucsv['ctau'] == 1) & (tree_mucsv['lxybin'],'neventsmc'].iloc[0]
    if sample == "bphi":
        x_data = bphiallcsv['mass'].tolist()
        y_data = bphiallcsv['ctau'].tolist()
        z_data_trigsfunc = bphiallcsv['trigsf_unc'].tolist()
        z_data_oneminuseff = bphiallcsv['oneminuseff'].tolist()

    sys_trigsfunc = Rbf(x_data, y_data, z_data_trigsfunc, epsilon=0.03)
    sys_oneminuseff = Rbf(x_data, y_data, z_data_oneminuseff, epsilon=0.03)

    return sys_trigsfunc(mass,ctau), sys_oneminuseff(mass,ctau)

# trigsf_unc_val, oneminuseff_val = get_systematics(mass = 2, ctau = 1, sample="hzd")   
# print "trigsfunc", float(trigsf_unc_val), "oneminuseff", float(oneminuseff_val) 

def get_mcstatunc(mass = 2, ctau = 1, sample="hzd", whichbin="lxybin1_ptbin1_isobin1"):
    
    if sample == "hzd":
        x_data = hzdbincsv['mass'].tolist()
        y_data = hzdbincsv['ctau'].tolist()
        z_data = hzdbincsv['nevt_'+whichbin].tolist()

    if sample == "bphi":
        x_data = bphibincsv['mass'].tolist()
        y_data = bphibincsv['ctau'].tolist()
        z_data = bphibincsv['nevt_'+whichbin].tolist()

    sys_mcstatunc = Rbf(x_data, y_data, z_data, epsilon=0.03)

    return 1/np.sqrt(np.clip(sys_mcstatunc(mass,ctau),1,None))

# mcstat_unc_val = get_mcstatunc(mass = 2, ctau = 1, sample="hzd", whichbin=getbin(inputcard=card))
# print "mcstatunc", float(mcstat_unc_val)

def seddatacard(inputcard = "card", mass = 2, ctau = 1, signalrate = 100,trigsf_unc = 0,oneminuseff = 0,mcstat_unc = 0 ):

    outputstr = (inputcard.split('mass{}_'.format(mass))[1]).split('_bernstein')[0]
    # print outputstr
    outputcard = "datacard_mass{}_ctau{}_".format(mass,ctau) + outputstr + ".txt"

    os.system("sed 's/rate\s100/rate {:.7f}/g' {} >  {}".format(signalrate,inputcard,outputcard))
    os.system("sed -i 's/trigsf_unc\slnN\s1/trigsf_unc lnN {:.3f}/g' {}".format(1+trigsf_unc,outputcard))
    os.system("sed -i 's/oneminuseff\slnN\s1/oneminuseff lnN {:.3f}/g' {}".format(1+oneminuseff,outputcard))
    os.system("sed -i '/^mcstat_unc/ s/lnN\s1/lnN {:.3f}/g' {} ".format(1+mcstat_unc,outputcard))
    # os.system("sed -i '/^pdf_index/d' {} ".format(outputcard))
    os.system("sed -i 's/^mean .*$/mean param {} {}/' {} ".format(mass, mass*10*1.1*0.01*0.01*0.5, outputcard))
    os.system("sed -i 's/^mean1 .*$/mean1 param {} {}/' {} ".format(mass, mass*10*1.1*0.01*0.01*0.5, outputcard))
    os.system("sed -i '/^bgnorm /d' {} ".format(outputcard))
    os.system("sed -i '/^signorm /d' {} ".format(outputcard))

# seddatacard(inputcard = card, mass = 2, ctau = 1, signalrate = 0.5,trigsf_unc = float(trigsf_unc_val),oneminuseff = float(oneminuseff_val),mcstat_unc = float(mcstat_unc_val) )

# exit()


dct = {}
  
for k in range(len(ctaus)):
    dct['ctau{}_exp50'.format(ctaus[k])] = []

for j in range(len(masses)):

    os.chdir("./mass_{}".format(masses[j]))

    dct['mass{}_exp50'.format(masses[j])] = []
    dct['mass{}_nevtexp50'.format(masses[j])] = []
    dct['mass{}_nevtup'.format(masses[j])] = []
    dct['mass{}_nevtdown'.format(masses[j])] = []
    dct['mass{}_nevtupup'.format(masses[j])] = []
    dct['mass{}_nevtdowndown'.format(masses[j])] = []
    dct['mass{}_nevtobs'.format(masses[j])] = []
    dct['mass{}_brfracexp50'.format(masses[j])] = []
    dct['mass{}_brfracup'.format(masses[j])] = []
    dct['mass{}_brfracdown'.format(masses[j])] = []
    dct['mass{}_brfracupup'.format(masses[j])] = []
    dct['mass{}_brfracdowndown'.format(masses[j])] = []
    dct['mass{}_brfracobs'.format(masses[j])] = []
    dct['mass{}_significance'.format(masses[j])] = []
    dct['mass{}_possignificance'.format(masses[j])] = []

    for k in range(len(ctaus)):

        print "looking at mass ", masses[j], "and ctau ", ctaus[k]

        print "The acceptance of this mass,ctau point is ", acceptance(mass=masses[j],ctau=ctaus[k],sample="hzd",whichbin="allbins")
        acc_allbins = acceptance(mass=masses[j],ctau=ctaus[k],sample="hzd",whichbin="allbins")

        
        lxybins = np.array([[0.0,0.2], [0.2,1.0], [1.0,2.4], [2.4,3.1], [3.1,7.0], [7.0,11.0]])
        ptbins = np.array([[0,25],[25,5000]])
        isobins = np.array([[1,0,0],[0,1,0],[0,0,1]])

        signal_rates = {}

        for l in range(len(lxybins)):
            for m in range(len(ptbins)):                                                      
                for n in range(len(isobins)):  

                    acc_val = acceptance(mass=masses[j],ctau=ctaus[k],sample="hzd",whichbin="lxybin{}_ptbin{}_isobin{}".format(l+1,m+1,n+1)) 

                    if  acc_val > 0.000000000001:
                        signal_rates['lxybin{}_ptbin{}_isobin{}'.format(l+1,m+1,n+1)] = acc_val
                    else:
                        signal_rates['lxybin{}_ptbin{}_isobin{}'.format(l+1,m+1,n+1)] = 1e-8

                    # signal_rates['lxybin{}_ptbin{}_isobin{}'.format(l+1,m+1,n+1)] = acceptance(2,1,sample="hzd",whichbin="lxybin{}_ptbin{}_isobin{}".format(l+1,m+1,n+1)) 
                    
        print "The sum of signal rates or total acceptance is", sum(signal_rates.values())

        if sum(signal_rates.values()) >= 0.01:
            for sr in signal_rates.keys():
                if signal_rates[sr] <= 1e-8: continue
                signal_rates[sr] = signal_rates[sr] * 10000.0
        elif sum(signal_rates.values()) < 0.01 and sum(signal_rates.values()) >= 0.001:
            for sr in signal_rates.keys():
                if signal_rates[sr] <= 1e-8: continue
                signal_rates[sr] = signal_rates[sr] * 100000.0
        elif sum(signal_rates.values()) < 0.001 and sum(signal_rates.values()) >= 0.0001:
            for sr in signal_rates.keys():
                if signal_rates[sr] <= 1e-8: continue
                signal_rates[sr] = signal_rates[sr] * 1000000.0
        elif sum(signal_rates.values()) < 0.0001:
            for sr in signal_rates.keys():
                if signal_rates[sr] <= 1e-8: continue
                signal_rates[sr] = signal_rates[sr] * 10000000.0

        # print "The scaled signal rates for lxy bins", signal_rates

        totalsignalrate = sum(signal_rates.values())

        print "The sum of scaled signal rates", sum(signal_rates.values())

        nevtdf = pd.read_csv('../csvlimits1/nevtUL_mass{}_v0.csv'.format(masses[j]))
        nevtexp = nevtdf.iloc[0]["ctau = {}".format(ctaus[k])]

        print "The nevt exp for this mass,ctau point is", nevtexp

        for sr in signal_rates.keys():
                if signal_rates[sr] <= 1e-8: continue
                signal_rates[sr] = signal_rates[sr] * (nevtexp/totalsignalrate)

        print "The sum of new scaled signal rates", sum(signal_rates.values())

        for file in glob.glob("simple*.txt".format(masses[j])):
            # print(file)
            bin_val = getbin(inputcard=file)
            signalrate_val = signal_rates[bin_val]
            trigsf_unc_val, oneminuseff_val = get_systematics(mass = masses[j], ctau = ctaus[k], sample="hzd")
            mcstat_unc_val = get_mcstatunc(mass = masses[j], ctau = ctaus[k], sample="hzd", whichbin=bin_val)
            seddatacard(inputcard = file, mass = masses[j], ctau = ctaus[k], signalrate = signalrate_val,trigsf_unc = trigsf_unc_val,oneminuseff = oneminuseff_val,mcstat_unc = mcstat_unc_val )

        # exit()

        os.system('combineCards.py -S datacard_mass{}_ctau{}_Lxy*.txt > datacard_mass{}_ctau{}_allbinsbias.txt'.format(masses[j],ctaus[k],masses[j],ctaus[k]))
                                            
                                        
        # os.system('rm datacard_mass{}_ctau{}_Lxy*.txt'.format(masses[j],ctaus[k]))


    os.chdir("./..")
 
