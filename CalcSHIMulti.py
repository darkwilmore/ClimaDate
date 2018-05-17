import sys
import pandas as pd
import json
import numpy as np
from scipy.stats import norm

def shi_gamma(df)

	ordered = df.sort_values(['Lat','Lon','Year','Month','Day'])  

	arr = np.asarray(ord	ordered['TmaxScaled'] = np.convolve(arr,np.ones(3,dtype=int),'same')

	ordered['Median']=ordered.groupby(['Month'])['TmaxScaled'].transform(lambda x :x.quantile(0.5) )

	dfR,dfL=ordered.loc[frame.TmaxScaled>frame.Median],frame.loc[frame.TmaxScaled<=frame.Median]

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
		ordered['stdR'] = np.where(df['Month']==count, param[count-1][0], np.NaN)
	
	for count in range(1,13):
		ordered['stdR'] = np.where(df['Month']==count, param[count-1][0], df['stdR'])
	
	for count in range(1,13):
		ordered['stdL'] = np.where(df['Month']==count, param[count-1][1], np.NaN)	

	for count in range(1,13):
		ordered['stdL'] = np.where(df['Month']==count, param[count-1][1], df['stdL'])	

	ordered['SHI'] = np.NaN

	for index,row in df.iterrows():
   		shi = split.psplitnorm(row['TmaxScaled'], mode=row['Median'], sd1=row['stdR'], sd2=row['stdL'])[0]
   		ordered.loc[index,'SHI'] = shi
    
	for index,row in df.iterrows():
  		shi = norm.ppf(row['SHI'])
  		ordered.loc[index,'SHI'] = shi
  		
  	return ordered