import os
import glob
import ROOT


def get_ssb(fname):
    srate = -1
    shapefname = None
    with open(fname) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("rate "):
                srate = float(line.split()[1])
            if line.startswith("shapes"):
                shapefname = line.split()[3]
                normstring = (line.split()[4]).split(":")[1]

    f = ROOT.TFile(shapefname)
    mybkgs = f.Get("mybkgs")
    mybkgs.cd()
    norm = ROOT.gDirectory.Get("{}".format(normstring))
    f.Close()
    bg = norm.getVal()

    return srate/max(bg,1e-6)**0.5

mass_ctaus = [
        [5.85, 1],
        ]

for mass, ctau in mass_ctaus:
    os.chdir("./mass_{}".format(mass))

    fnames = glob.glob("datacard_mass*_ctau{}_Lxy*.txt".format(ctau))

    top5bins = sorted(fnames, key=get_ssb)[-5:]
    print("top 5 bins",top5bins)

    cards_to_run = []
    for i,fname in enumerate(top5bins):
        os.system("combineCards.py -S {} > top5bins_mass{}_ctau{}_bin{}.txt".format(fname, mass, ctau, i+1))
        cards_to_run.append("top5bins_mass{}_ctau{}_bin{}.txt".format(mass, ctau, i+1))
    os.system("combineCards.py -S {} > top5bins_mass{}_ctau{}_allbins.txt".format(" ".join(top5bins), mass, ctau))
    cards_to_run.append("top5bins_mass{}_ctau{}_allbins.txt".format(mass, ctau))

    for cardname in cards_to_run:

        postfix = cardname.split("_",1)[1].rsplit(".",1)[0]

        cmd = "combine -M FitDiagnostics --cminDefaultMinimizerStrategy 0 --rMin 0 --saveShapes --saveWithUncertainties  --saveOverallShapes --savePredictionsPerToy {} ; mv fitDiagnostics.root fitDiagnostics_{}.root".format(cardname, postfix)
        print("Running:", cmd)
        os.system(cmd)

        cmd = "combine -M AsymptoticLimits {} > limits_{}.log".format(cardname, postfix)
        print("Running:", cmd)
        os.system(cmd)

        cmd = "combine -M Significance --uncapped 1 --rMin -5 {} > significance_{}.log".format(cardname, postfix)
        print("Running:", cmd)
        os.system(cmd)

    os.chdir("./..")
