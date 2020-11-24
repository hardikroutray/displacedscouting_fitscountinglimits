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

x = []
y = []
zexp50 = []
accbr = []
acc = []
acc_lxybin0 = []
acc_lxybin1 = []
acc_lxybin2 = []
acc_lxybin3 = []
acc_lxybin4 = []
acc_lxybin5 = []


# masseslist = [0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7,0.725,0.75,0.775,0.8,0.825,0.85,0.875,0.9,0.925,0.95,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.25,4.5,4.75,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

masseslist = [8, 9]

# masseslist = [0.2, 0.202, 0.204, 0.206, 0.208, 0.21, 0.212, 0.214, 0.216, 0.218, 0.22, 0.222, 0.224, 0.226, 0.228, 0.23, 0.232, 0.234, 0.236, 0.238, 0.24, 0.242, 0.244, 0.246, 0.248, 0.25, 0.252, 0.254, 0.256, 0.258, 0.26, 0.262, 0.264, 0.266, 0.268, 0.27, 0.272, 0.274, 0.276, 0.278, 0.28, 0.282, 0.284, 0.286, 0.288, 0.29, 0.292, 0.294, 0.296, 0.298, 0.3, 0.302, 0.304, 0.306, 0.308, 0.31, 0.312, 0.314, 0.316, 0.318, 0.32, 0.322, 0.324, 0.326, 0.328, 0.33, 0.332, 0.334, 0.336, 0.338, 0.34, 0.342, 0.344, 0.346, 0.348, 0.35, 0.352, 0.354, 0.356, 0.358, 0.36, 0.362, 0.364, 0.366, 0.368, 0.37, 0.372, 0.374, 0.376, 0.378, 0.38, 0.382, 0.384, 0.386, 0.388, 0.39, 0.392, 0.394, 0.396, 0.398, 0.4, 0.402, 0.404, 0.406, 0.408, 0.41, 0.412, 0.414, 0.416, 0.418, 0.42, 0.422, 0.424, 0.426, 0.428, 0.43, 0.432, 0.434, 0.436, 0.438, 0.44, 0.442, 0.444, 0.446, 0.448, 0.45, 0.452, 0.454, 0.456, 0.458, 0.46, 0.462, 0.464, 0.466, 0.468, 0.47, 0.472, 0.474, 0.476, 0.478, 0.48, 0.482, 0.484, 0.486, 0.48, 0.49, 0.492, 0.494, 0.496, 0.498,0.5,0.505,0.51,0.515,0.52,0.525,0.53,0.535,0.54,0.545,0.55,0.56,0.565,0.57,0.575,0.58,0.585,0.59,0.595,0.6,0.605,0.61,0.615,0.62,0.625,0.63,0.635,0.64,0.645,0.65,0.655,0.66,0.665,0.67,0.675,0.68,0.685,0.69,0.695,0.7,0.705,0.71,0.715,0.72,0.725,0.73,0.735,0.74,0.745,0.75,0.755,0.76,0.765,0.77,0.775,0.78,0.785,0.79,0.795,0.8,0.805,0.81,0.815,0.82,0.825,0.83,0.835,0.84,0.845,0.85,0.855,0.86,0.865,0.87,0.875,0.88,0.885,0.89,0.895,0.9,0.905,0.91,0.915,0.92,0.925,0.93,0.935,0.94,0.945,0.95,0.955,0.96,0.965,0.97,0.975,0.98,0.985,0.99,0.995,1,1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09,1.1,1.11,1.12,1.13,1.14,1.15,1.16,1.17,1.18,1.19,1.2,1.21,1.22,1.23,1.24,1.25,1.26,1.27,1.28,1.29,1.3,1.31,1.32,1.33,1.34,1.35,1.36,1.37,1.38,1.39,1.4,1.41,1.42,1.43,1.44,1.45,1.46,1.47,1.48,1.49,1.5,1.51,1.52,1.53,1.54,1.55,1.56,1.57,1.58,1.59,1.6,1.61,1.62,1.63,1.64,1.65,1.66,1.67,1.68,1.69,1.7,1.71,1.72,1.73,1.74,1.75,1.76,1.77,1.78,1.79,1.8,1.81,1.82,1.83,1.84,1.85,1.86,1.87,1.88,1.89,1.9,1.91,1.92,1.93,1.94,1.95,1.96,1.97,1.98,1.99,2,2.02,2.04,2.06,2.08,2.1,2,12,2.14,2.16,2.18,2.20,2.22,2.24,2.26,2.28,2.3,2.32,2.34,2.36,2.38,2.40,2.42,2.44,2.46,2.48,2.5,2.52,2.54,2.56,2.58,2.6,2.62,2.64,2.66,2.68,2.70,2.72,2.74,2.76,2.78,2.8,2.82,2.84,2.86,2.88,2.9,2.92,2.94,2.96,2.98,3, 3.03, 3.06, 3.09, 3.12, 3.15, 3.18, 3.21, 3.24, 3.27, 3.3, 3.33, 3.36, 3.39, 3.42, 3.45, 3.48, 3.51, 3.54, 3.57, 3.6, 3.63, 3.66, 3.69, 3.71, 3.75, 3.78, 3.81, 3.84, 3.87, 3.9, 3.92, 3.96, 3.99, 4, 4.04, 4.08, 4.12, 4.16, 4.2, 4.24, 4.28, 4.32, 4.36, 4.4, 4.44, 4.48, 4.52, 4.56, 4.6, 4.64, 4.68, 4.72, 4.76, 4.8, 4.84, 4.88, 4.92, 4.96, 5, 5.05, 5.1, 5.15, 5.2, 5.25, 5.3, 5.35, 5.4, 5.45, 5.5, 5.55, 5.6, 5.65, 5.7, 5.75, 5.8, 5.85, 5.9, 5.95, 6, 6.06, 6.12, 6.18, 6.24, 6.3, 6.36, 6.42, 6.48, 6.54, 6.6, 6.66, 6.72, 6.78, 6.84, 6.9, 6.96, 7, 7.07, 7.14, 7.21, 7.28, 7.35, 7.42, 7.49, 7.56, 7.63, 7.7, 7.77, 7.84, 7.91, 7.98, 8, 8.08, 8.16, 8.24, 8.32, 8.4, 8.48, 8.56, 8.64, 8.72, 8.8, 8.88, 8.96, 9, 9.09, 9.18, 9.27, 9.36, 9.45, 9.54, 9.63, 9.72, 9.81, 9.9, 9.99, 10, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15,  15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 16, 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7, 16.8, 16.9, 17, 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7, 17.8, 17.9, 18, 18.1, 18.2, 18.3, 18.4, 18.5, 18.6, 18.7, 18.8, 18.9, 19, 19.1, 19.2, 19.3, 19.4, 19.5, 19.6, 19.7, 19.8, 19.9, 20, 20.2, 20.4, 20.6, 20.8, 21, 21.2, 21.4, 21.6, 21.8, 22, 22.2, 22.4, 22.6, 22.8, 23, 23.2, 23.4, 23.6, 23.8, 24, 24.2, 24.4, 24.6, 24.8, 25]



print len(masseslist)

masses = [masseslist[int(sys.argv[1])]]

# ctaus = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1,2,3,4,5,6,7,8,9,10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

ctaus = [5,10,50,100]



dct = {}
dct1 = {}

for k in range(len(ctaus)):
    dct['ctau{}_exp50'.format(ctaus[k])] = []

for j in range(len(masses)):

    os.chdir("./mass_{}".format(masses[j]))

    dct['mass{}_exp50'.format(masses[j])] = []

    for k in range(len(ctaus)):

        print "looking at mass ", masses[j], "and ctau ", ctaus[k]

        from acc_fit_vf import acceptance
        acceptancearray = acceptance(masses[j],ctaus[k],sample = "hzd")
        # print acceptancearray
        
        acc_lxyall = acceptancearray[0]
        # br_zdtomumu = br
        print "The acceptance of this mass,ctau point is ", acc_lxyall
        
        g = []
        signal_rates = []    
        lxybins = np.array([[0.0,0.2], [0.2,1.0], [1.0,2.4], [2.4,3.1], [3.1,7.0], [7.0,11.0]])

        for l in range(len(lxybins)):

            if  acceptancearray[l+1] > 0.000000000001:
                signal_rates.append(acceptancearray[l+1])
            else:
                signal_rates.append(0.00000001)

        print "The signal rates for lxy bins", signal_rates

        acc_lxybin0.append(signal_rates[0])
        acc_lxybin1.append(signal_rates[1])
        acc_lxybin2.append(signal_rates[2])
        acc_lxybin3.append(signal_rates[3])
        acc_lxybin4.append(signal_rates[4])
        acc_lxybin5.append(signal_rates[5])

        print "The sum of signal rates or total acceptance is", sum(signal_rates)

        if sum(signal_rates) >= 0.01:
            signal_rates = [sr * 10000.0 for sr in signal_rates]
        elif sum(signal_rates) < 0.01 and sum(signal_rates) >= 0.001:
            signal_rates = [sr * 100000.0 for sr in signal_rates]
        elif sum(signal_rates) < 0.001 and sum(signal_rates) >= 0.0001:
            signal_rates = [sr * 1000000.0 for sr in signal_rates]
        elif sum(signal_rates) < 0.0001:
            signal_rates = [sr * 10000000.0 for sr in signal_rates]

        print "The scaled signal rates for lxy bins", signal_rates
        totalsignalrate = sum(signal_rates)

        print "The sum of scaled signal rates", sum(signal_rates)

        for file in glob.glob("simple*.txt"):
            # print(file)
            if "Lxy0.0_0.2_bernstein" in file:
                os.system("sed 's/rate\s100/rate {}/g' {} >  datacard_mass{}_ctau{}_Lxy0.0_0.2.txt".format(signal_rates[0],file,masses[j],ctaus[k]))
            elif "Lxy0.0_0.2_counting" in file:
                os.system("sed 's/rate\s100/rate {}/g' {} >  datacard_mass{}_ctau{}_Lxy0.0_0.2.txt".format(signal_rates[0]*0.9244,file,masses[j],ctaus[k]))
            elif "Lxy0.2_1.0_bernstein" in file:
                os.system("sed 's/rate\s100/rate {}/g' {} >  datacard_mass{}_ctau{}_Lxy0.2_1.0.txt".format(signal_rates[1],file,masses[j],ctaus[k]))
            elif "Lxy0.2_1.0_counting" in file:
                os.system("sed 's/rate\s100/rate {}/g' {} >  datacard_mass{}_ctau{}_Lxy0.2_1.0.txt".format(signal_rates[1]*0.9244,file,masses[j],ctaus[k]))
            elif "Lxy1.0_2.4_bernstein" in file:
                os.system("sed 's/rate\s100/rate {}/g' {} >  datacard_mass{}_ctau{}_Lxy1.0_2.4.txt".format(signal_rates[2],file,masses[j],ctaus[k]))
            elif "Lxy1.0_2.4_counting" in file:
                os.system("sed 's/rate\s100/rate {}/g' {} >  datacard_mass{}_ctau{}_Lxy1.0_2.4.txt".format(signal_rates[2]*0.9244,file,masses[j],ctaus[k]))
            elif "Lxy2.4_3.1_bernstein" in file:
                os.system("sed 's/rate\s100/rate {}/g' {} >  datacard_mass{}_ctau{}_Lxy2.4_3.1.txt".format(signal_rates[3],file,masses[j],ctaus[k]))
            elif "Lxy2.4_3.1_counting" in file:
                os.system("sed 's/rate\s100/rate {}/g' {} >  datacard_mass{}_ctau{}_Lxy2.4_3.1.txt".format(signal_rates[3]*0.9244,file,masses[j],ctaus[k]))
            elif "Lxy3.1_7.0_bernstein" in file:
                os.system("sed 's/rate\s100/rate {}/g' {} >  datacard_mass{}_ctau{}_Lxy3.1_7.0.txt".format(signal_rates[4],file,masses[j],ctaus[k]))
            elif "Lxy3.1_7.0_counting" in file:
                os.system("sed 's/rate\s100/rate {}/g' {} >  datacard_mass{}_ctau{}_Lxy3.1_7.0.txt".format(signal_rates[4]*0.9244,file,masses[j],ctaus[k]))
            elif "Lxy7.0_11.0_bernstein" in file:
                os.system("sed 's/rate\s100/rate {}/g' {} >  datacard_mass{}_ctau{}_Lxy7.0_11.0.txt".format(signal_rates[5],file,masses[j],ctaus[k]))
            elif "Lxy7.0_11.0_counting" in file:
                os.system("sed 's/rate\s100/rate {}/g' {} >  datacard_mass{}_ctau{}_Lxy7.0_11.0.txt".format(signal_rates[5]*0.9244,file,masses[j],ctaus[k]))

        os.system('combineCards.py -S datacard_mass{}_ctau{}_Lxy*.txt > datacard_mass{}_ctau{}_alllxy.txt'.format(masses[j],ctaus[k],masses[j],ctaus[k]))
        os.system('combine -M  AsymptoticLimits -m {} --rAbsAcc=0.0001 --rRelAcc=0.001 datacard_mass{}_ctau{}_alllxy.txt > com.out'.format(masses[j],masses[j],ctaus[k]))
        os.system('cat com.out')                                                                                                            
                                        
        os.system('rm datacard_mass{}_ctau{}_Lxy*.txt'.format(masses[j],ctaus[k]))


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
        


        os.system('combine -M Significance --uncapped 1 --rMin=-5 datacard_mass{}_ctau{}_alllxy.txt > com_1.out'.format(masses[j],ctaus[k]))
        os.system('cat com_1.out')                                                                                            

        com1_out = open('com_1.out','r') 

        for line in com1_out:                                                                                                           

            if line[:13] == 'Significance:':     
                                                                                               
                coms_exp = float(line[13:])   


        # exp_xsec = coms_exp
        exp_xsec = coml_exp*totalsignalrate
        # exp_xsec = (coml_exp*totalsignalrate)/(10.1*acc_lxyall)

        print "The expected 50% UL xsec is ", exp_xsec

        x.append(masses[j])
        y.append(ctaus[k])
        zexp50.append(exp_xsec)
        acc.append(acc_lxyall)

        dct['mass{}_exp50'.format(masses[j])].append(exp_xsec)
        dct['ctau{}_exp50'.format(ctaus[k])].append(exp_xsec)
        
    print "the expected 50% UL xsec for this mass for different ctaus is", dct['mass{}_exp50'.format(masses[j])]
    os.chdir("./..")
 
# print "the expected 50% UL xsec for ctau 20 for different masses is", dct['ctau20_exp50']

print "The ordered mass of all points", x
print "The ordered ctau of all points", y
print "The ordered acceptance of all points",acc
print "The ordered exp 50% UL of all points", zexp50

print "The ordered acceptance of all points in lxybin0",acc_lxybin0
print "The ordered acceptance of all points in lxybin1",acc_lxybin1
print "The ordered acceptance of all points in lxybin2",acc_lxybin2
print "The ordered acceptance of all points in lxybin3",acc_lxybin3
print "The ordered acceptance of all points in lxybin4",acc_lxybin4
print "The ordered acceptance of all points in lxybin5",acc_lxybin5

# print dct['mass{}_exp50'.format(masses[0])]

import pandas as pd

arr = []
arr.append(masses[0])
for i in range(len(ctaus)):
    arr.append(dct['mass{}_exp50'.format(masses[0])][i])
# arr = ["{},{},{},{},{}".format(mass,num[0],num[1],num[2],num[3]),columns]                                                                                                         
print arr

# df = pd.DataFrame([arr],columns=['mass', 'ctau = 0.1', 'ctau = 0.2', 'ctau = 0.3', 'ctau = 0.4', 'ctau = 0.5', 'ctau = 0.6', 'ctau = 0.7', 'ctau = 0.8', 'ctau = 0.9', 'ctau = 1', 'ctau = 2',  'ctau = 3',  'ctau = 4', 'ctau = 5',  'ctau = 6', 'ctau = 7', 'ctau = 8', 'ctau = 9', 'ctau = 10', 'ctau = 15', 'ctau = 20', 'ctau = 25', 'ctau = 30', 'ctau = 35', 'ctau = 40', 'ctau = 45', 'ctau = 50', 'ctau = 55', 'ctau = 60', 'ctau = 65', 'ctau = 70', 'ctau = 75', 'ctau = 80', 'ctau = 85', 'ctau = 90', 'ctau = 95', 'ctau = 100'])

df = pd.DataFrame([arr],columns=['mass', 'ctau = 5', 'ctau = 10', 'ctau = 50', 'ctau = 100'])

print df

if not os.path.exists("csvlimits"):                                                                               
    os.makedirs("csvlimits")

df.to_csv('csvlimits/NevtUL_mass{}_v0.csv'.format(masses[0]),index=False)


