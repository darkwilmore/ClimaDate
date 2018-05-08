import pandas as pd
import numpy as np
from scipy.stats import norm
from rpy2.robjects.packages import importr

split = importr('fanplot')

df = pd.read_csv("/Users/marcoventurini/Documents/spark-2.0.0-bin-hadoop2.7/data/DailyMax_lat-14_lon35.csv")

df['Median']=df.groupby(['Month'])['TmaxScaled'].transform(lambda x :x.quantile(0.5) )

dfR,dfL=df.loc[df.TmaxScaled>df.Median],df.loc[df.TmaxScaled<=df.Median]

vectL = []
for count in range (1,13):
     vectL.append(np.r_[dfL.TmaxScaled[dfL.Month==count],dfL.Median[dfL.Month==count]*2 - dfL.TmaxScaled[dfL.Month==count]])
     
	
vectR = []	
for count in range (1,13):
	  vectR.append(np.r_[dfR.TmaxScaled[dfR.Month==count],dfR.Median[dfR.Month==count]*2-dfR.TmaxScaled[dfR.Month==count]])
	  
	  
	  
paramL = []	  
for array in vectL:
	paramL.append(norm.fit(array))



paramR = []	  
for array in vectR:
	paramR.append(norm.fit(array))



param = []	
for count in range(0,12):
	param.append((paramL[count][1],paramR[count][1],paramR[count][0]))

for count in range(1,13):
	df['stdR'] = np.where(df['Month']==count, param[count-1][0], np.NaN)
	
for count in range(1,13):
	df['stdR'] = np.where(df['Month']==count, param[count-1][0], df['stdR'])
	
for count in range(1,13):
	df['stdL'] = np.where(df['Month']==count, param[count-1][1], np.NaN)	

for count in range(1,13):
	df['stdL'] = np.where(df['Month']==count, param[count-1][1], df['stdL'])	

df['SHI'] = np.NaN

for index,row in df.iterrows():
    shi = split.psplitnorm(row['TmaxScaled'], mode=row['Median'], sd1=row['stdR'], sd2=row['stdL'])[0]
    df.loc[index,'SHI'] = shi
    
for index,row in df.iterrows():
    shi = norm.ppf(row['SHI'])
    df.loc[index,'SHI'] = shi
    
df.to_csv('/Users/marcoventurini/Documents/spark-2.0.0-bin-hadoop2.7/SHIshort.csv',columns=['Year','Month','Day','SHI'],index=False)
