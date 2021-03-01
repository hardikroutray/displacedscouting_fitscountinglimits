import os
import sys
import ROOT
import glob

import os
import ROOT
import numpy as np

dotop5 = True

masses = [5.85]
mass = masses[int(sys.argv[1])]
mass1 = mass
ctaus = [1]


def pullplots(filename="fitDiagnostics.root", what = "postfit_sb", dotop5 = False, ch = "ch4"):
        fname = filename
        mass = mass1

        f = ROOT.TFile(fname)
        
        if what == "postfit_sb":
                directory = "shapes_fit_s"
        elif what == "postfit_b":
                directory = "shapes_fit_b"
        elif what == "prefit":
                directory = "shapes_prefit"


        ## single bin                                                                           
        data = ROOT.gDirectory.Get("{}/total_data".format(directory))
        bkg = ROOT.gDirectory.Get("{}/total_background".format(directory))
        bkgsig = ROOT.gDirectory.Get("{}/total_overall".format(directory))
        sig = ROOT.gDirectory.Get("{}/total_signal".format(directory))

        ## single channel in multichannel
        if dotop5:
                if not ROOT.gDirectory.Get("{}/{}/data".format(directory,ch)):
                        print("s+b fit does not exist")
                        return

                data = ROOT.gDirectory.Get("{}/{}/data".format(directory,ch))
                bkg = ROOT.gDirectory.Get("{}/{}/total_background".format(directory,ch))        
                bkgsig = ROOT.gDirectory.Get("{}/{}/total".format(directory,ch))
                sig = ROOT.gDirectory.Get("{}/{}/total_signal".format(directory,ch))           

        mlow = mass*(1-5*0.011)
        mhigh = mass*(1+5*0.011)
        bw = (mhigh-mlow)/100.


        hdata = ROOT.TH1F("hdata", "data", 100, mlow,mhigh)
        hdata.SetLineColor(ROOT.kBlack)
        hdata.SetLineWidth(2)
        hdata.SetMarkerSize(5)

        hbg = ROOT.TH1F("hbg", "bg", 100, mlow,mhigh)
                
        if what == "postfit_b":
                hbg.SetLineColor(ROOT.kBlue)
        if what == "postfit_sb" or "prefit":
                hbg.SetLineColor(ROOT.kRed)

        hbg.SetLineWidth(3)

        hsig = ROOT.TH1F("hsig", "sig", 100, mlow,mhigh)
        hsig.SetLineColor(ROOT.kGreen)
        hsig.SetLineWidth(3)

        hsb = ROOT.TH1F("hsb", "sig+bg", 100, mlow,mhigh)
        hsb.SetLineColor(ROOT.kBlue)
        hsb.SetLineWidth(3)

        pull = ROOT.TH1F("pull", "", 100, mlow,mhigh)
        pull.SetLineColor(ROOT.kRed)
        pull.SetLineWidth(3)

        for ibin in range(1,101):
                mult = bw
                hdata.SetBinContent(ibin,data.GetY()[ibin-1]*mult )
                hbg.SetBinContent(ibin, bkg.GetBinContent(ibin)*mult)
                hbg.SetBinError(ibin, 0.) # since they are NaN                                  

                hsig.SetBinContent(ibin, sig.GetBinContent(ibin)*mult)
                hsb.SetBinContent(ibin, bkgsig.GetBinContent(ibin)*mult)

                # https://github.com/aminnj/plottery/blob/master/plottery.py#L603              
                numer_val = data.GetY()[ibin-1]*mult
                numer_err = numer_val**0.5
                denom_val = bkg.GetBinContent(ibin)*mult
                denom_err = 1e-6
                ratio_val = numer_val / denom_val

                # gaussian pull                                                              
                pull_val = (ratio_val-1.)/((numer_err**2.+denom_err**2.)**0.5)
                if numer_val > 1e-6:
                        # more correct pull, but is inf when 0 data, so fall back to gaus pull in that case      
                        # pull_val = ROOT.RooStats.NumberCountingUtils.BinomialObsZ(numer_val,denom_val,denom_err/denom_val);                                                                                    
                        pull_val = ROOT.RooStats.NumberCountingUtils.BinomialObsZ(numer_val,denom_val,1e-6);

                pull.SetBinContent(ibin,pull_val)

        ratio = hdata.Clone("ratio")
        ratio.Divide(hbg)
        ratio.SetLineColor(ROOT.kRed)
        ratio.SetLineWidth(3)

        hdata.SetMinimum(0)
        hbg.SetMinimum(0)
        hsig.SetMinimum(0)
        hsb.SetMinimum(0)
        ratio.SetMinimum(0.)
        ratio.SetMaximum(2.)
        pull.SetMinimum(-4)
        pull.SetMaximum(4)
        
        c1 = ROOT.TCanvas()
        # pad1 = ROOT.TPad("pad1", "The pad 80% of the height", 0.0, 0.2, 1.0, 1.0, 0)          
        # pad2 = ROOT.TPad("pad2", "The pad 20% of the height", 0.0, 0.0, 1.0, 0.2, 0)        
        pad1 = ROOT.TPad("pad1", "The pad 80% of the height", 0.0, 0.4, 1.0, 1.0, 0)
        pad2 = ROOT.TPad("pad2", "The pad 20% of the height", 0.0, 0.2, 1.0, 0.4, 0)
        pad3 = ROOT.TPad("pad3", "The pad 20% of the height", 0.0, 0.0, 1.0, 0.2, 0)
        c1.cd()

        pad1.Draw()
        pad2.Draw()
        pad3.Draw()

        pad1.cd()
        pad1.SetTickx()
        pad1.SetTicky()
        pad1.SetBottomMargin(0.04)

        x = ROOT.RooRealVar("x","x",mlow,mhigh)
        l = ROOT.RooArgList(x)
        xframe = x.frame(ROOT.RooFit.Title("{}   {}".format(outname1,what)))
        data1 = ROOT.RooDataHist("data", "data", l, hdata)
        # bkg1 = ROOT.RooDataHist("bkg", "bkg", l, hbg)                                         
        data1.plotOn(xframe,ROOT.RooFit.Name("data"))
        # bkg1.plotOn(xframe,ROOT.RooFit.MarkerSize(0),ROOT.RooFit.LineColor(2),ROOT.RooFit.Name("bkg"))
        xframe.Draw()
        xframe.GetYaxis().SetTitle("Events/ {} GeV".format(bw))
        xframe.GetYaxis().SetTitleSize(0.05)
        xframe.GetYaxis().SetLabelSize(0.045)
        xframe.GetYaxis().SetTitleOffset(0.95)


        hbg.Draw("L SAME")
        if what == "postfit_sb" or what == "prefit":
                hsig.Draw("L SAME")
        if what == "postfit_sb" or what == "prefit":
                hsb.Draw("L SAME")

        leg1 = ROOT.TLegend()
        leg1.SetLineColor(0)
        leg1.SetFillColor(0)
        leg1.SetFillStyle(0)
        leg1.AddEntry(xframe.findObject("data"), "Data", "pe")
        # leg1.AddEntry(hdata)                                                                           
        leg1.AddEntry(hbg)
        if what == "postfit_sb" or what == "prefit":
                leg1.AddEntry(hsig)
        if what == "postfit_sb" or what == "prefit":
                leg1.AddEntry(hsb)


        leg1.SetTextFont(42)
        leg1.SetBorderSize(0)
        leg1.Draw()
        
        xframe1 = x.frame(ROOT.RooFit.Title(" "))
        xframe1.GetXaxis().SetLabelSize(0.17)
        xframe1.GetYaxis().SetLabelSize(0.15)
        xframe1.GetXaxis().SetTitleSize(0.21)
        xframe1.GetYaxis().SetTitleSize(0.15)
        xframe1.GetXaxis().SetTitleOffset(0.85)
        xframe1.GetYaxis().SetTitleOffset(0.28)
        # xframe1.GetXaxis().SetTitle("Dimuon Mass [GeV]")                                      
        # xframe1.GetYaxis().SetTitle("#scale[1.3]{#frac{data - fit}{#sigma_{data}}}")          
        xframe1.GetYaxis().SetTitle("Data/Bkg")
        xframe1.GetYaxis().SetLabelSize(0.15)
        xframe1.SetMinimum(0.5)
        xframe1.SetMaximum(1.5)
        xframe1.GetYaxis().SetNdivisions(5)

        pad2.cd()
        pad2.SetTickx()
        pad2.SetTicky()
        # pad2.SetGridy()     
                                                                           
        pad2.SetTopMargin(0.18)
        pad2.SetBottomMargin(0.2)
        xframe1.Draw()
        ratio.Draw("PE SAME")          

        xframe2 = x.frame(ROOT.RooFit.Title(" "))
        xframe2.GetXaxis().SetLabelSize(0.17)
        xframe2.GetYaxis().SetLabelSize(0.15)
        xframe2.GetXaxis().SetTitleSize(0.21)
        xframe2.GetYaxis().SetTitleSize(0.15)
        xframe2.GetXaxis().SetTitleOffset(0.85)
        xframe2.GetYaxis().SetTitleOffset(0.28)
        xframe2.GetXaxis().SetTitle("Dimuon Mass [GeV]")
        xframe2.GetYaxis().SetTitle("#scale[1.3]{#frac{data - fit}{#sigma_{data}}}")
        # xframe2.GetYaxis().SetTitle("Data/Bkg")                                              
        xframe2.GetYaxis().SetLabelSize(0.15)
        xframe2.SetMinimum(-5)
        xframe2.SetMaximum(5)
        xframe2.GetYaxis().SetNdivisions(5)

        pad3.cd()
        pad3.SetTickx()
        pad3.SetTicky()
        # pad3.SetGridy()                                                                       

        pad3.SetTopMargin(0.0)
        pad3.SetBottomMargin(0.4)
        xframe2.Draw()
        pull.Draw("HIST SAME")
        pull.SetMarkerStyle(20)
        pull.SetMarkerSize(0.8)

        ROOT.gStyle.SetOptStat(0)

        if dotop5 == False:
                c1.Draw()
                c1.SaveAs("combinefits1/{}_{}.png".format(outname1,what))
        else:
                c1.Draw()
                c1.SaveAs("combinefits_top5/{}_{}.png".format(outname1,what))
        



print "Running on mass", mass

os.chdir("mass_{}".format(mass))

for j in range(len(ctaus)):

        print "Looking at ctau----------",ctaus[j], "----------------"
        nfile = 0

        if dotop5:

                if not os.path.exists("combinefits_top5"):
                        os.mkdir("combinefits_top5")


                for i in range(5):
                        outname1 = "top5_ch{}".format(i+1)
                        pullplots(filename = "fitDiagnostics_mass{}_ctau{}_allbins.root".format(mass,ctaus[j]), what = "postfit_sb", dotop5 = True, ch = "ch{}".format(i+1))
                        pullplots(filename = "fitDiagnostics_mass{}_ctau{}_allbins.root".format(mass,ctaus[j]), what = "postfit_b", dotop5 = True, ch = "ch{}".format(i+1))
                        pullplots(filename = "fitDiagnostics_mass{}_ctau{}_allbins.root".format(mass,ctaus[j]), what = "prefit", dotop5 = True, ch = "ch{}".format(i+1))

                exit()
 
        for file in glob.glob("datacard_mass{}_ctau{}_Lxy*.txt".format(mass,ctaus[j])):
                nfile+=1
                # if "ctau{}_".format(ctaus[j]) in file:
                #         print(file)
                cardname = file
                outname = (cardname.split("datacard_mass{}_".format(mass,ctaus[j]))[1]).split(".txt")[0]
                # if outname != "ctau1_Lxy3.1_7.0_pt25_Inf_2isomu":
                #         continue

                # if "0.0_0.2" in outname:
                #         continue
                # if "0.2_1.0" in outname:
                #         continue
                # if "0.0_0.2" in outname:
                #         continue
                # if "1.0_2.4" in outname:
                #         continue
                # if "2.4_3.1" in outname:
                #         continue

                images = glob.glob("combinefits1/*png")
                if any(outname in s for s in images):
                        continue

                print outname

                outname1 = (cardname.split("datacard_".format(mass,ctaus[j]))[1]).split(".txt")[0]
                # print outname1

                if not os.path.exists("combinefits1"):
                        os.mkdir("combinefits1")


                os.system("combine -M FitDiagnostics {} --cminDefaultMinimizerStrategy 0 --saveShapes --saveWithUncertainties --saveOverallShapes".format(cardname))

                pullplots(filename = "fitDiagnostics.root", what = "postfit_sb")
                pullplots(filename = "fitDiagnostics.root", what = "postfit_b")
                pullplots(filename = "fitDiagnostics.root", what = "prefit")

                # if nfile == 5:
                #         exit()

        print "number of files processed" , nfile
