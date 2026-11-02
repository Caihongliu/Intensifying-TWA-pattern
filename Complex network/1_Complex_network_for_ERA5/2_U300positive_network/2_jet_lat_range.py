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
import seaborn as sns
from scipy.linalg import norm
from scipy import stats
from scipy.stats.mstats import ttest_ind
import cartopy
import cartopy.crs as ccrs
from scipy.signal import detrend 
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
from scipy.stats import gaussian_kde

def runavg(x, width):
    # 输入一原始时间序列和宽度，返回一个平滑以后的时间序列
    n = len(x)
    x = np.append(x, np.append(x, x))
    x_smooth = np.convolve(x, np.ones(width)/width, mode='valid')
    xs = x_smooth[n:2*n]
    return xs



# # ------------------------------------------------------------ # #
# # --------------------    read networks    ------------------- # #
ds0=xr.open_dataset("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/Function/U300Positive_significant_995.nc")
lat=ds0.lat1; lon=ds0.lon1
network0=ds0.significant_99
print(np.shape(network0))
print(lat)
networks0 = np.where(network0 >= 20.0, 1.0, 0.0) # 又是零一矩阵
dim_0 = np.shape(networks0)
# print(networks0)
 
# # ---------------------------- remove abs(lon_diff > 50°)  ---------------------------- # #

# for ilon in range(180):
#     for jlon in range(180):
#         if (abs(jlon - ilon) >25 ):
#             networks0[:,ilon,:,jlon]=0
        

# # ---------------------------- keep   30<lat<40  ---------------------------- # #


for ilat in range(0,45+11):
    networks0[ilat,:,:,:] = np.where((((ilat-11)*2+1)>=0) & (((ilat-11)*2+1)<=30),networks0[ilat,:,:,:] , 0.0)


# # ----------------------------lat_tile 变成正确的经纬度信息 ---------------------------- # #

lat_diff =np.zeros((56,180,56,180))
for ilat in range(-11+11,45+11):
    for jlat in range(-11+11,45+11):
        lat_diff[ilat,:,jlat,:] = ((jlat-11)*2+1) - ((ilat-11)*2+1)


# # ----------------------------lon_tile ---------------------------- # #

lon_diff =np.zeros((56,180,56,180))
for ilon in range(180):
    for jlon in range(180):
        lon_diff[:,ilon,:,jlon] = (jlon*2+1) - (ilon*2+1)

print(len(np.unique(lon_diff)))
print(len(np.unique(lat_diff)))

# # ---------------------------- calculate pdf ---------------------------- # #
# # 4个点算一个
pdf = np.zeros((21,41))
pdf_all = np.zeros((21,41))
lat_interval = np.linspace(-40, 40, 21); print(lat_interval) #真正经纬度
lon_interval = np.linspace(-80, 80, 41); print(lon_interval)

 
for i in range(21):
    print("i = "+str(i)) # 把4°以内的加起来

    lat_min = lat_interval[i] - 2.0
    lat_max = lat_interval[i] + 2.0


    for j in range(41):
        print("j = "+str(j))

        lon_min = lon_interval[j] - 2.0
        lon_max = lon_interval[j] + 2.0

        # Apply element-wise logical conditions
        condition = (lat_diff >= lat_min) & (lat_diff <= lat_max) & (lon_diff >= lon_min) & (lon_diff <= lon_max)
        networks0_new = np.where(condition, networks0, 0.0)

 

        pdf_all[i,j] = np.sum(networks0)
        pdf[i,j] = np.sum(networks0_new)

    
        

# pdf = pdf/np.sum(pdf)
print("----  heat-U100Positive,  concurrent days > 99%  ----")
print(pdf)

##--##--##-- 存储 --##--##--##
pdf = xr.DataArray(data= pdf, dims=['lat' ,'lon'],
            coords= {'lat':np.linspace(-40, 40, 21),'lon':np.linspace(-80, 80, 41)})

pdf_all = xr.DataArray(data= pdf_all, dims=['lat' ,'lon'],
            coords= {'lat':np.linspace(-40, 40, 21),'lon':np.linspace(-80, 80, 41)})

 
ds3 = xr.Dataset(data_vars= dict(pdf = pdf,pdf_all = pdf_all ))
ds3.to_netcdf("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/2_U300positive_network/PDF/0_30_PDF_U300_positive.nc")
ds3.close()
