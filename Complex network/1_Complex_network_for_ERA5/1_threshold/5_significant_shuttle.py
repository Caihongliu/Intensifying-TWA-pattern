from cartopy.util import add_cyclic_point
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from copy import copy
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import shapely.geometry as sgeom
import matplotlib.colorbar as colorbar
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from matplotlib.patches import Rectangle
import warnings
import matplotlib.colors as mcolors
warnings.filterwarnings("ignore")
import matplotlib as mpl
mpl.use("Agg")
from cartopy.util import add_cyclic_point#去除投影中间白线
from scipy.linalg import norm
from scipy import stats
from scipy.stats.mstats import ttest_ind
from scipy.signal import detrend
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
import random

def runavg(x, width):
 #输入一原始时间序列和宽度，返回一个平滑以后的时间序列
    n = len(x)
    x = np.append(x, np.append(x, x))
    x_smooth = np.convolve(x, np.ones(width)/width, mode='valid')
    xs = x_smooth[n:2*n]
    return xs

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma



##--##--##--##--  创建 Heatwave 数据  --##--##--##--##
ds1 = xr.open_dataset("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/1_threshold/T_01_day_JA_79_23.nc")
heatwave = ds1.heatwave.loc[:,:,-21:89,:]
lat = ds1.lat.loc[-21:89]
lon = ds1.lon

ds2 = xr.open_dataset("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/1_threshold/Westerly_01_day_JA_79_23.nc")
westerly_day_data = ds2.westerly_day_data.loc[:,:,-21:89,:]



## ------------- 3.各点 急流天数和热浪天数

# 每个点的急流天数
point_sum_westerly_day =  np.zeros((np.shape(heatwave)[2],np.shape(heatwave)[3]))
point_sum_westerly_day = np.sum(westerly_day_data[:,:,:,:], axis=(0,1))
print(point_sum_westerly_day)


# 每个点的热浪天数
point_sum_hw_day =  np.zeros((np.shape(heatwave)[2],np.shape(heatwave)[3]))
point_sum_hw_day = np.sum(heatwave[:,:,:,:], axis=(0,1))
print(point_sum_hw_day)

## ------------- 4.制作阈值表  打乱1000次的99th 阈值

max_hw=int(np.max(point_sum_hw_day)) ; print(max_hw) ;#1015   #667
max_west=np.max(point_sum_westerly_day) ; print(max_west) ;#820  #547
 
min_hw=int(np.min(point_sum_hw_day)) ; print(min_hw) ;#313     #6
min_west=np.min(point_sum_westerly_day) ; print(min_west) ;#549   #355
 
##--##--##--##--   计算  --##--##--##--##
matrix995 = np.zeros((1101,1101), dtype=np.int16)  ## 99% 阈值


for iA in range(10,1101,10):
    if iA%50==0: 
        print("iA"+str(iA))
        ia=iA
        for iB in range(10,1101,10):
            print("iB"+str(iB))
            ib=iB

        
            checklist =np.zeros((1000), dtype=np.int16)## 重复 1000 次

            for k in range(1000):
                PC1=np.zeros((45*62), dtype=np.int16) #热浪
                PC2=np.zeros((45*62), dtype=np.int16) #急流
                
                PC1[:ia]=1
                PC2[:ib]=1 
                random.shuffle(PC1)
                random.shuffle(PC2)

                checklist[k] = np.sum(np.array(PC1+PC2==2), axis=0)


            threshold995= int(np.ceil(np.percentile(checklist, 99.5 )))
            del(checklist)

            # --------------------------------
            # 向上取整：
            # 0-10   -> 10
            # 11-20  -> 20
            # 21-30  -> 30
            # ...
            # --------------------------------

            if ia == 10:
                ia_start = 0
            else:
                ia_start = ia - 9

            if ib == 10:
                ib_start = 0
            else:
                ib_start = ib - 9

            ia_end = min(ia, 1100)
            ib_end = min(ib, 1100)

            matrix995[ia_start:ia_end + 1, ib_start:ib_end + 1] = threshold995

       
 
##--##--##-- 存储 --##--##--##
matrix995 = xr.DataArray(data= matrix995, dims=['events1', 'events2'],
            coords= {'events1':range(1101) , 'events2':range(1101) })

ds3 = xr.Dataset(data_vars= dict(matrix995 = matrix995))
ds3.to_netcdf("/public/home/fcai/abc/2Paper_jet/Fig2/1_threshold/45y_62d_matrix99.5_network_per10.nc")
ds3.close()

## ------------- 4.各点 急流天数和热浪天数 打乱1000次的99th 阈值

# A=np.zeros((45*92)) #热浪
# B=np.zeros((45*92, 45,180)) #急流
# count =np.zeros((1000, 45,180, 45,180))

# for ilat in range(45):
#     print("ilat"+str(ilat))
#     for ilon in range(180):
#         print("ilon"+str(ilon))

        
#         hw_events=point_sum_hw_day[ilat,ilon] #固定一个点
#         westerly_events=point_sum_westerly_day[:,:]
        
#         k = 0
#         while(k < 1000):
#             A=0
#             B=0
#             A[0:hw_events]=1
#             B[0:westerly_events+1,:,:]=1 
#             random.shuffle(A)
#             random.shuffle(B)

            
#             A_tile_3D = np.tile(A, (45,180)).transpose(2,0,1)
#             count[k,ilat,ilon,:,:] = np.sum(A_tile_3D+B[:,:,:]==2, axis=0)
#             k=k+1


# count99 = np.percentile(count, 99, axis=0 )

