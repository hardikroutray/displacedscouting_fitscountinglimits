import ROOT
import pandas as pd
import csv
import os
import sys
from array import array

import matplotlib.pyplot as plt
import numpy as np
import glob
from matplotlib.colors import LogNorm

from ipywidgets import *
import numpy, scipy, scipy.optimize
import matplotlib
from mpl_toolkits.mplot3d import  Axes3D
from matplotlib import cm # to colormap 3D surfaces from blue to red                                                                                                                 
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
#%matplotlib notebook                                                                                                                                                                
graphWidth = 800 # units are pixels                                                                                                                                                  
graphHeight = 600 # units are pixels                                                                                                                                                 

# 3D contour plot lines                                                                                                                                                              
numberOfContourLines = 16


x = []
y = []
accbr = []
acc = []
trigsf_unc = []
oneminuseff = []
# acc_lxybin0 = []
# acc_lxybin1 = []
# acc_lxybin2 = []
# acc_lxybin3 = []
# acc_lxybin4 = []
# acc_lxybin5 = []
signal_rates_arr = []

    
if sys.argv[1] == "bphi":
    print("running on bphi sample")
    masses = [0.5, 0.6, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4]
elif sys.argv[1] == "hzd":
    print("running on hzd sample")
    masses = [0.5, 0.6, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 25]

ctaus = [0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]

# masses = [0.5,2,5,10]
# ctaus = [0.5,1,50,100]

tree_muMC = ROOT.TChain('t')

if sys.argv[1] == "bphi":
    tree_muMC.Add("./bphi_mass_ctau_scan.root")
elif sys.argv[1] == "hzd":
    tree_muMC.Add("./hzd_mass_ctau_scan.root")

if sys.argv[1] == "bphi":
    tree_mucsv = pd.read_csv('info_bphi.csv')
elif sys.argv[1] == "hzd":
    tree_mucsv = pd.read_csv('info_hzd.csv')

for j in range(len(masses)):

    for k in range(len(ctaus)):

        print ("looking at mass ", masses[j], "and ctau ", ctaus[k])

        h = ROOT.TH1F("h","h", 1000, 0, 100)

        if ctaus[k] == 100:
            tree_muMC.Draw('mass>>h','trigsf*(sample_mass == {} && sample_ctau == {})'.format(masses[j],ctaus[k]),'goff')
        elif ctaus[k] > 10 and ctaus[k] < 100:
            tree_muMC.Draw('mass>>h','trigsf*(100/{})*exp((10*ct/100) - (10*ct/{}))*(sample_mass == {} && sample_ctau == {})'.format(ctaus[k],ctaus[k],masses[j],100),'goff')
        elif ctaus[k] == 10:
            tree_muMC.Draw('mass>>h','trigsf*(sample_mass == {} && sample_ctau == {})'.format(masses[j],ctaus[k]),'goff')
        elif ctaus[k] > 1 and ctaus[k] < 10:
            tree_muMC.Draw('mass>>h','trigsf*(10/{})*exp((10*ct/10) - (10*ct/{}))*(sample_mass == {} && sample_ctau == {})'.format(ctaus[k],ctaus[k],masses[j],10),'goff')
        elif ctaus[k] == 1:
            tree_muMC.Draw('mass>>h','trigsf*(sample_mass == {} && sample_ctau == {})'.format(masses[j],ctaus[k]),'goff')
        elif ctaus[k] > 0 and ctaus[k] < 1:
            tree_muMC.Draw('mass>>h','trigsf*(1/{})*exp((10*ct/1) - (10*ct/{}))*(sample_mass == {} && sample_ctau == {})'.format(ctaus[k],ctaus[k],masses[j],1),'goff')

        if ctaus[k] == 100:
            evt = 0
            for event in tree_muMC:
                if event.sample_mass == masses[j] and event.sample_ctau ==  ctaus[k]:
                    if evt != 0:
                        break
                    total = event.nevents_input
                    # total = event.nevents_input*0.5
                    # br = event.br_zdtomumu
                    trigsf_error = event.trigsf_error
                    oneminuseffval = event.oneminuseff
                    print (total)
                    evt+=1

        elif ctaus[k] > 10 and ctaus[k] < 100:
            evt = 0
            for event in tree_muMC:
                if event.sample_mass == masses[j] and event.sample_ctau ==  100:
                    if evt != 0:
                        break
                    total = event.nevents_input
                    # total = event.nevents_input*0.5
                    # br = event.br_zdtomumu
                    trigsf_error = event.trigsf_error
                    oneminuseffval = event.oneminuseff
                    print (total)
                    evt+=1

        elif ctaus[k] == 10:
            evt = 0
            for event in tree_muMC:
                if event.sample_mass == masses[j] and event.sample_ctau ==  ctaus[k]:
                    if evt != 0:
                        break
                    total = event.nevents_input
                    # total = event.nevents_input*0.5
                    # br = event.br_zdtomumu
                    trigsf_error = event.trigsf_error
                    oneminuseffval = event.oneminuseff
                    print (total)
                    evt+=1

        elif ctaus[k] > 1 and ctaus[k] < 10:
            evt = 0
            for event in tree_muMC:
                if event.sample_mass == masses[j] and event.sample_ctau ==  10:
                    if evt != 0:
                        break
                    total = event.nevents_input
                    # total = event.nevents_input*0.5
                    # br = event.br_zdtomumu         
                    trigsf_error = event.trigsf_error
                    oneminuseffval = event.oneminuseff
                    print (total)
                    evt+=1

        elif ctaus[k] == 1:
            evt = 0
            for event in tree_muMC:
                if event.sample_mass == masses[j] and event.sample_ctau ==  ctaus[k]:
                    if evt != 0:
                        break
                    total = event.nevents_input
                    # total = event.nevents_input*0.5
                    # br = event.br_zdtomumu         
                    trigsf_error = event.trigsf_error
                    oneminuseffval = event.oneminuseff
                    print (total)
                    evt+=1


        elif ctaus[k] > 0 and ctaus[k] < 1:
            evt = 0
            for event in tree_muMC:
                if event.sample_mass == masses[j] and event.sample_ctau ==  1:
                    if evt != 0:
                        break
                    total = event.nevents_input
                    # total = event.nevents_input*0.5
                    # br = event.br_zdtomumu         
                    trigsf_error = event.trigsf_error
                    oneminuseffval = event.oneminuseff
                    print (total)
                    evt+=1


        acceptance = h.Integral()/total
        print ("The acceptance of this mass,ctau point is ", acceptance)

        
        # g = []
        # signal_rates = []

        signal_rates = {}
        signal_rates['mass'] = masses[j]
        signal_rates['ctau'] = ctaus[k]

        lxybins = np.array([[0.0,0.2], [0.2,1.0], [1.0,2.4], [2.4,3.1], [3.1,7.0], [7.0,11.0]])
        ptbins = np.array([[0,25],[25,5000]])
        isobins = np.array([[2],[1],[0]])

        for l in range(len(lxybins)):
            for m in range(len(ptbins)):
                for n in range(len(isobins)):

                    # g.append(ROOT.TH1F("g[{}]".format(l),"g[{}]".format(j), 1000, 0, 100))
                    g = ROOT.TH1F("g","g", 1000, 0, 100)
                    
                    if ctaus[k] == 100:
                        tree_muMC.Draw('mass>>g','trigsf*(sample_mass == {} && sample_ctau == {} && lxy > {} && lxy < {} && dimuon_pt > {} && dimuon_pt < {} && nisomu == {})'.format(masses[j],ctaus[k],lxybins[l,0], lxybins[l,1], ptbins[m,0], ptbins[m,1], isobins[n,0]),'goff')
                    elif ctaus[k] > 10 and ctaus[k] < 100:
                        tree_muMC.Draw('mass>>g','trigsf*(100/{})*exp((10*ct/100) - (10*ct/{}))*(sample_mass == {} && sample_ctau == {} && lxy > {} && lxy < {} && dimuon_pt > {} && dimuon_pt < {} && nisomu == {})'.format(ctaus[k],ctaus[k],masses[j],100,lxybins[l,0], lxybins[l,1], ptbins[m,0], ptbins[m,1], isobins[n,0]),'goff')
                    elif ctaus[k] == 10:
                        tree_muMC.Draw('mass>>g','trigsf*(sample_mass == {} && sample_ctau == {} && lxy > {} && lxy < {} && dimuon_pt > {} && dimuon_pt < {} && nisomu == {})'.format(masses[j],ctaus[k],lxybins[l,0],lxybins[l,1], ptbins[m,0], ptbins[m,1], isobins[n,0]),'goff')
                    elif ctaus[k] > 1 and ctaus[k] < 10:
                        tree_muMC.Draw('mass>>g','trigsf*(10/{})*exp((10*ct/10) - (10*ct/{}))*(sample_mass == {} && sample_ctau == {} && lxy > {} && lxy < {} && dimuon_pt > {} && dimuon_pt < {} && nisomu == {})'.format(ctaus[k],ctaus[k],masses[j],10,lxybins[l,0], lxybins[l,1], ptbins[m,0], ptbins[m,1], isobins[n,0]),'goff')
                    elif ctaus[k] == 1:
                        tree_muMC.Draw('mass>>g','trigsf*(sample_mass == {} && sample_ctau == {} && lxy > {} && lxy < {} && dimuon_pt > {} && dimuon_pt < {} && nisomu == {})'.format(masses[j],ctaus[k],lxybins[l,0],lxybins[l,1], ptbins[m,0], ptbins[m,1], isobins[n,0]),'goff')
                    elif ctaus[k] > 0 and ctaus[k] < 1:
                        tree_muMC.Draw('mass>>g','trigsf*(1/{})*exp((10*ct/1) - (10*ct/{}))*(sample_mass == {} && sample_ctau == {} && lxy > {} && lxy < {} && dimuon_pt > {} && dimuon_pt < {} && nisomu == {})'.format(ctaus[k],ctaus[k],masses[j],1,lxybins[l,0],lxybins[l,1], ptbins[m,0], ptbins[m,1], isobins[n,0]),'goff')

                    # signal_rates.append(g[l].Integral())
                    signal_rates['acc_lxybin{}_ptbin{}_isobin{}'.format(l+1,m+1,n+1)] = g.Integral()/total

                    if (ctaus[k] > 10 and ctaus[k] <= 100): 

                        nevt = tree_mucsv.loc[(tree_mucsv['mass'] == masses[j]) & (tree_mucsv['ctau'] == 100) & (tree_mucsv['lxybin'] == l+1) & (tree_mucsv['ptbin'] == m+1) & (tree_mucsv['nisomu'] == isobins[n,0]),'neventsmc'].iloc[0]

                    elif (ctaus[k] > 1 and ctaus[k] <= 10): 

                        nevt = tree_mucsv.loc[(tree_mucsv['mass'] == masses[j]) & (tree_mucsv['ctau'] == 10) & (tree_mucsv['lxybin'] == l+1) & (tree_mucsv['ptbin'] == m+1) & (tree_mucsv['nisomu'] == isobins[n,0]),'neventsmc'].iloc[0]

                    elif (ctaus[k] > 0 and ctaus[k] <= 1): 

                        nevt = tree_mucsv.loc[(tree_mucsv['mass'] == masses[j]) & (tree_mucsv['ctau'] == 1) & (tree_mucsv['lxybin'] == l+1) & (tree_mucsv['ptbin'] == m+1) & (tree_mucsv['nisomu'] == isobins[n,0]),'neventsmc'].iloc[0]
                    
                    signal_rates['nevt_lxybin{}_ptbin{}_isobin{}'.format(l+1,m+1,n+1)] = nevt
                    
                    g.Reset()


        signal_rates_arr.append(signal_rates)
        
        # print ("The signal rates for lxy bins", signal_rates)

        # acc_lxybin0.append(signal_rates[0]/total)
        # acc_lxybin1.append(signal_rates[1]/total)
        # acc_lxybin2.append(signal_rates[2]/total)
        # acc_lxybin3.append(signal_rates[3]/total)
        # acc_lxybin4.append(signal_rates[4]/total)
        # acc_lxybin5.append(signal_rates[5]/total)

        x.append(masses[j])
        y.append(ctaus[k])
        acc.append(acceptance)

        trigsf_unc.append(trigsf_error)
        oneminuseff.append(oneminuseffval)

        # accbr.append(acceptance*br)

        h.Reset()

print ("The ordered mass of all points", x)
print ("The ordered ctau of all points", y)
print ("The ordered acceptance of all points",acc)

# print ("The ordered acceptance of all points in lxybin0",acc_lxybin0)
# print ("The ordered acceptance of all points in lxybin1",acc_lxybin1)
# print ("The ordered acceptance of all points in lxybin2",acc_lxybin2)
# print ("The ordered acceptance of all points in lxybin3",acc_lxybin3)
# print ("The ordered acceptance of all points in lxybin4",acc_lxybin4)
# print ("The ordered acceptance of all points in lxybin5",acc_lxybin5)

acc_fit = open("acc_fit_vf_test.py", "a")
acc_fit.write("mass_hzd = {}\n".format(x))
acc_fit.write("ctau_hzd = {}\n".format(y))
acc_fit.write("acchzd = {}\n".format(acc))
# acc_fit.write("acchzd_lxybin0 = {}\n".format(acc_lxybin0))
# acc_fit.write("acchzd_lxybin1 = {}\n".format(acc_lxybin1))
# acc_fit.write("acchzd_lxybin2 = {}\n".format(acc_lxybin2))
# acc_fit.write("acchzd_lxybin3 = {}\n".format(acc_lxybin3))
# acc_fit.write("acchzd_lxybin4 = {}\n".format(acc_lxybin4))
# acc_fit.write("acchzd_lxybin5 = {}\n".format(acc_lxybin5))
acc_fit.close()


df1 =  pd.DataFrame({'mass':x,'ctau':y,'acc_allbins':acc,'trigsf_unc':trigsf_unc,'oneminuseff':oneminuseff},columns=['mass','ctau','acc_allbins','trigsf_unc','oneminuseff'])
df1.to_csv('totalacceptance_{}.csv'.format(sys.argv[1]),index=False)

df2 = pd.DataFrame(signal_rates_arr)
df2.to_csv('binbybinacceptance_{}.csv'.format(sys.argv[1]),index=False)


# dw=pd.DataFrame( [[20, 30, {"ab":"1", "we":"2", "as":"3"},"String"]],
#                 columns=['ColA', 'ColB', 'ColC', 'ColdD'])
# pd.concat([dw.drop(['ColC'], axis=1), dw['ColC'].apply(pd.Series)], axis=1)


def spline(data):
    f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)

    matplotlib.pyplot.grid(True)
    axes = Axes3D(f)

    x_data = data[0]
    y_data = data[1]
    z_data = data[2]

    xModel = numpy.linspace(min(x_data), max(x_data), 20)
    yModel = numpy.linspace(min(y_data), max(y_data), 20)
    X, Y = numpy.meshgrid(xModel, yModel)

    rbf = Rbf(x_data, y_data, z_data, epsilon=0.03)
    Z = rbf(X, Y)

    axes.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=1, antialiased=True)

    axes.scatter(x_data, y_data, z_data) # show data along with plotted surface                                                                                                     
    axes.set_title('Surface Plot (click-drag with mouse)') # add a title for surface plot                                                                                           
    axes.set_xlabel('mass') # X axis data label                                                                                                                                     
    axes.set_ylabel('ctau(in mm)') # Y axis data label                                                                                                                              
    axes.set_zlabel('acceptance') # Z axis data label                                                                                                                               

    plt.show()
    plt.close('all') # clean up after using pyplot or else thaere can be memory and process problems       


def splinevalue(x,y):
    x_data = data[0]
    y_data = data[1]
    z_data = data[2]

    rbf = Rbf(x_data, y_data, z_data, epsilon=0.03)
    return rbf(x,y)   
    
'''
data = [x, y, acc]
spline(data)

modelPredictions = splinevalue(data[0],data[1])
absError = modelPredictions - acc

SE = numpy.square(absError) # squared errors                                                                                                                                     
MSE = numpy.mean(SE) # mean squared errors                                                                                                                                       
RMSE = numpy.sqrt(MSE) # Root Mean Squared Error, RMSE                                                                                                                           
Rsquared = 1.0 - (numpy.var(absError) / numpy.var(acc))
print('RMSE:', RMSE)
print('R-squared:', Rsquared)

print(splinevalue(5,50))
'''
