import sys
sys.path.insert(0,"/Users/marcoventurini/Downloads/indices_rc1")		
import compute
import indices as ind
import pandas as pd
import json
import numpy as np
from scipy.stats import norm

df = pd.read_csv('/Users/marcoventurini/Documents/spark-2.0.0-bin-hadoop2.7/data/MonthlyPrp_lat-14_lon35.csv')
arrayD = np.asarray(df.PrpSummed)

arraySPI3 = ind.spi_gamma(arrayD,3)
df['SPI3'] = np.NaN
for index,row in df.iterrows():
    df.loc[index,'SPI3'] = arraySPI3[index]
    
arraySPI12 = ind.spi_gamma(arrayD,12)
df['SPI12'] = np.NaN
for index,row in df.iterrows():
    df.loc[index,'SPI12'] = arraySPI12[index]
    
df.to_csv('/Users/marcoventurini/Documents/spark-2.0.0-bin-hadoop2.7/SPIshort.csv',columns=['Year','Month','SPI3','SPI12'],index=False)
