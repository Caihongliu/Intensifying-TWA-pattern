from cartopy.util import add_cyclic_point
import numpy as np
import xarray as xr
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
 #输入一原始时间序列和宽度，返回一个平滑以后的时间序列
    n = len(x)
    x = np.append(x, np.append(x, x)) # 数据扩展三倍，用于循环边界处理
    x_smooth = np.convolve(x, np.ones(width)/width, mode='valid')
    xs = x_smooth[n:2*n] # 取出中间部分（原始数据对应的滑动结果）
    # 调整中心对齐
    offset = (width - 1) // 2
    xs = x_smooth[n - offset : 2 * n - offset]
    return xs

# # ------------------------------------------------------------ # #
# # --------------------    read networks    ------------------- # #
ds0=xr.open_dataset("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/Function/U300Positive_significant_995.nc")
lat=ds0.lat1; lon=ds0.lon1;print(ds0)
network0=ds0.significant_99
print(np.shape(network0))

networks0 = np.where(network0 >= 20.0, 1.0, 0.0) # 又是零一矩阵
dim_0 = np.shape(networks0)
# print(networks0)

# # ---------------------------- remove abs(lon_diff > 50°)  ---------------------------- # #

for ilon in range(180):
    for jlon in range(180):
        if (abs(jlon - ilon) >25 ):
            networks0[:,ilon,:,jlon]=0

# # ---------------------------- keep   30<lat<90  ---------------------------- # #

# 0<lat<90
for ilat in range(-11,45):
    networks0[ilat,:,:,:] = np.where(((ilat*2+1)>=0) ,networks0[ilat,:,:,:] , 0.0)


# for ilat in range(-11,45):
#     networks0[ilat+11,:,:,:] = np.where(((ilat*2+1)>=30) ,networks0[ilat+11,:,:,:] , 0.0)



# # ---------------------------- tile ---------------------------- # #

lat_diff =np.zeros((56,180,56,180))
for ilat in range(-11,45):
    for jlat in range(-11,45):
        lat_diff[ilat+11,:,jlat+11,:] = (jlat) - (ilat)





# # ---------------------------- calculate pdf ---------------------------- # #
pdf = np.zeros((89))
for i in range(89):
    
    networks0_new = np.where(lat_diff==(i-44), networks0, 0.0)
    pdf[i] = np.sum(networks0_new)
    print(" i = "+str(i)+",    lat diff= "+str(i-44)+",    pdf = "+ str(pdf[i]))


pdf = pdf/np.sum(pdf)
print("----  heat-U100Positive,  concurrent days > 99%  ----")
print(pdf)

##--##--##-- 存储 --##--##--##
pdf = xr.DataArray(data= pdf, dims=['lat1' ],
            coords= {'lat1':range(89)})

print(pdf)
ds3 = xr.Dataset(data_vars= dict(pdf = pdf))
ds3.to_netcdf("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/2_U300positive_network/PDF_U300_postive.nc")
ds3.close()