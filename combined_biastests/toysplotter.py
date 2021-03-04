import os
import ROOT


mass = 12
ctau = 1
rinj = 2
batch = 1

os.chdir("mass_{}".format(mass))

fname1 = "fitDiagnostics0{}_{}_analysis{}.000000.root".format(batch,ctau,rinj)
fname2 = "higgsCombine0{}_{}_analysis{}.000000.GenerateOnly.mH{}.{}.root".format(batch,ctau,rinj,mass,batch)

channel = 1

f = ROOT.TFile(fname1)
t = f.Get("tree_fit_sb")
rows = []
for irow,row in enumerate(t):
    if irow < 4: continue
    r = row.r
    rLoErr = row.rLoErr
    rHiErr = row.rHiErr
    err = (r<0)*rHiErr+(r>0)*rLoErr
    err = max(err, 1e-6)
    bias = (r-rinj)/(err)
    fitstatus = row.fit_status
    bgvals = [getattr(row, "n_exp_final_binch{}_proc_background_{}".format(channel,i)) for i in range(1,101)]
    # print(bgvals)
    sigvals = [getattr(row, "n_exp_binch{}_proc_signal_{}".format(channel,i)) for i in range(1,101)]
    # print(bgvals, sigvals)

    nbg = getattr(row, "n_exp_final_binch{}_proc_background".format(channel))
    nsig = getattr(row, "n_exp_binch{}_proc_signal".format(channel))

    rows.append(dict(
        bias=bias,
        r=r,
        rLoErr=rLoErr,
        rHiErr=rHiErr,
        status=fitstatus,
        bgvals=bgvals,
        sigvals=sigvals,
        nbg=nbg,
        nsig=nsig,
        ))


f = ROOT.TFile(fname2)
os.system("mkdir -p plots/batch{}_ch{}".format(batch,channel))

mlow = mass*(1-5*0.011)
mhigh = mass*(1+5*0.011)

bw = (mhigh-mlow)/100.

for i,row in zip(range(1,60), rows):
# for i,row in zip(range(1,2), rows):
    if i%10 == 0: print(i)
    toy = ROOT.gDirectory.Get("toys/toy_{}".format(i))
    if not toy: break
    c1 = ROOT.TCanvas()
    x = ROOT.RooRealVar("x","x",mlow,mhigh)
    xframe = x.frame(ROOT.RooFit.Title("bias={:.3f}, rmeas={:.3f}, rLoErr={:.3g}, rHiErr={:.3g}, status={}".format(
        row["bias"],
        row["r"],
        row["rLoErr"],
        row["rHiErr"],
        row["status"],
        )))
    toy.plotOn(xframe)
    # xframe.GetYaxis().SetRangeUser(0.,8.)
    # xframe.GetYaxis().SetRangeUser(20000,25000)
    xframe.Draw()


    hbg = ROOT.TH1F("hbg", "bg", 100, mlow,mhigh)
    hbg.SetLineColor(ROOT.kRed)
    hbg.SetLineWidth(3)

    # hbg.GetYaxis().SetRangeUser(0.,8.0)



    hsig = ROOT.TH1F("hsig", "sig", 100, mlow,mhigh)
    hsig.SetLineColor(ROOT.kBlue)
    hsig.SetLineWidth(1)

    hsb = ROOT.TH1F("hsb", "sig+bg", 100, mlow,mhigh)
    hsb.SetLineColor(ROOT.kBlue)
    hsb.SetLineWidth(3)

    for ibin in range(1,101):
        v1 = row["bgvals"][ibin-1]
        if v1 == -999: continue
        hbg.SetBinContent(ibin, v1*bw)
        v2 = row["sigvals"][ibin-1]
        hsig.SetBinContent(ibin, v2*bw)
        hsb.SetBinContent(ibin, (v1+v2)*bw)

    hbg.Draw("same")
    # hsig.Draw("same")
    hsb.Draw("same")

    # print(row["sigvals"])
    # print(row["nsig"])

    # if hbg.Integral():
    #     hbg.Scale(row["nbg"]/hbg.Integral())
    #     hsig.Scale(abs(row["nsig"])/abs(hsig.Integral()))

    # hbg.Scale(0.0045) # bin width
    # hsig.Scale(0.0045) # bin width

    leg1 = ROOT.TLegend(0.7,0.7,0.9,0.9)
    leg1.SetLineColor(0)
    leg1.SetFillColor(0)
    leg1.SetFillStyle(0)
    leg1.AddEntry(xframe.findObject("x"), "Data", "pe")
    leg1.AddEntry(hbg)
    # leg1.AddEntry(hsig)
    leg1.AddEntry(hsb)
    leg1.SetTextFont(42)
    leg1.SetBorderSize(0)
    leg1.Draw()


    fname = "plots/batch{}_channel{}/toy_{}.png".format(batch,channel,i)
    c1.SaveAs(fname)

    # os.system("ic "+fname)
    # break

os.chdir("./..")
