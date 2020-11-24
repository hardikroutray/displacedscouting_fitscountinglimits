import pandas as pd
import glob

files = glob.glob("csvlimits/NevtUL*v0.csv") 
# files.sort()
# df = pd.concat((pd.read_csv(f, header = 0) for f in files))
# df.to_csv("numevt.csv")

df_list = []
for filename in sorted(files, key=lambda x: float(x.partition('_mass')[2].partition('_v0')[0])):
    df_list.append(pd.read_csv(filename))
full_df = pd.concat(df_list)

print full_df

# sum_column = full_df.loc[:,"nsideband==0":].sum(axis=0)
# print (sum_column)

full_df.to_csv('NevtUL_v0.csv',index=False)

