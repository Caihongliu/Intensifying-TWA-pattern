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
from scipy import signal
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
from global_land_mask import globe
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
#-*- coding:utf-8 –*-

def trend_result(data_3D,nt,nlat,nlon):
    ##--##--## 计算趋势 --##--##--##
    ngrd=nlat*nlon
    ano_grd = data_3D.values.reshape((nt, ngrd), order='F')
    x = np.linspace(1, nt, nt)

    # 建立斜率和显著性数组
    ano_rate = np.empty((ngrd,))
    p_values = np.empty((ngrd,))

    ano_rate[:] = np.nan
    p_values[:] = np.nan


    # 计算每个网格点的线性趋势和显著性
    for i in range(ngrd):
        slope, _, _, p_value, _ = stats.linregress(x, ano_grd[:, i])
        ano_rate[i] = slope
        p_values[i] = p_value


    # 将斜率和显著性结果重新调整为原来的网格形状
    day_shape = (nlat, nlon)
    anom_grd = ano_rate.reshape(day_shape, order='F') * 10
    p_values_grd = p_values.reshape(day_shape, order='F')
    return anom_grd,p_values_grd

def trend_1D_result(data_1D,nt):
    slope, _, _, _, _ = stats.linregress(np.linspace(1, nt, nt), data_1D)
    slope = slope*10
    return slope 


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def find_max_patch(data,lon1, lat_vals, lat_win=6, lon_win=20):
    max_val = -np.inf
    best_patch = None

    
    for lat_start in range(min(lat_vals), max(lat_vals) - lat_win + 2, 2):
        for lon_start in range((lon1-40), (lon1+40) - lon_win + 2, 2):

            lat_slice = np.arange(lat_start, lat_start + lat_win+2,2)
            lon_slice = np.arange(lon_start % 360, (lon_start + lon_win+2) % 360,2)

            # print(lat_slice);print(lon_slice)
            patch = data.sel(lat2=lat_slice, lon2=lon_slice)
            patch_mean = patch.mean().item()
            
            if patch_mean > max_val:
                max_val = patch_mean
                best_patch_lat_slice=lat_slice
                best_patch_lon_slice=lon_slice
    # print(best_patch_lat_slice)
    # print(best_patch_lon_slice)
    # print(max_val)
                
    return best_patch_lat_slice,best_patch_lon_slice , max_val

def mask_land(ds, label='land', lonname='lon'):
    if lonname == 'lon':
        lat = ds.lat.data
        lon = ds.lon.data
        if np.any(lon > 180):
            lon = lon - 180
            lons, lats = np.meshgrid(lon, lat)
            mask = globe.is_ocean(lats, lons)
            temp = []
            temp = mask[:, 0:(len(lon) // 2)].copy()
            mask[:, 0:(len(lon) // 2)] = mask[:, (len(lon) // 2):]
            mask[:, (len(lon) // 2):] = temp
        else:
            lons, lats = np.meshgrid(lon, lat)# Make a grid
            mask = globe.is_ocean(lats, lons)# Get whether the points are on ocean.
        ds.coords['mask'] = (('lat', 'lon'), mask)
    elif lonname == 'longitude':
        lat = ds.latitude.data
        lon = ds.longitude.data
        if np.any(lon > 180):
            lon = lon - 180
            lons, lats = np.meshgrid(lon, lat)
            mask = globe.is_ocean(lats, lons)
            temp = []
            temp = mask[:, 0:(len(lon) // 2)].copy()
            mask[:, 0:(len(lon) // 2)] = mask[:, (len(lon) // 2):]
            mask[:, (len(lon) // 2):] = temp
        else:
            lons, lats = np.meshgrid(lon, lat)
            mask = globe.is_ocean(lats, lons)
        lons, lats = np.meshgrid(lon, lat)
        mask = globe.is_ocean(lats, lons)
        ds.coords['mask'] = (('latitude', 'longitude'), mask)
    if label == 'land':
        ds = ds.where(ds.mask == True)
    elif label == 'ocean':
        ds = ds.where(ds.mask == False)
    return ds
## -------------1. 指数趋势
## -------------a. 对应 U 风 average change

#--##--##--##-- 创建 U250 数据  --##--##--##--##

ds4 = xr.open_dataset("/public/data2/model/fcai/data0/ERA5_ua_daily2x2/ua.1979.nc")
u = ds4.u.loc[:,300,:,:]
LAT = u.lat ; LON = u.lon 
print(np.shape(u)) 
u250 = np.zeros((45,62,np.shape(u)[1],np.shape(u)[2]))
 
for iyear in range(45):
  print('iyear = ',iyear+1979)
  zJuly = xr.open_dataset("/public/data2/model/fcai/data0/ERA5_ua_daily2x2/ua."+str(iyear+1979)+".nc")
  u250[iyear,0:31,:,:] = zJuly.u.loc[181:212,300,:,:] 
  zAugust = xr.open_dataset("/public/data2/model/fcai/data0/ERA5_ua_daily2x2/ua."+str(iyear+1979)+".nc")
  u250[iyear,31:62,:,:] = zAugust.u.loc[212:243,300,:,:]
  zJuly.close(); zAugust.close()


ua_year_run_90 = np.mean(u250[:,:,:,:],1)
u_JJA_clm = np.mean(ua_year_run_90[2:32,:,:],0)
tile_u_JJA_clm =np.tile(u_JJA_clm ,(45,1,1))
u_ano = np.subtract(ua_year_run_90[:,:,:],tile_u_JJA_clm)

u_ano = xr.DataArray(data= u_ano, dims=['time','lat','lon' ],
    coords= {'time':range(45),'lat':LAT.values ,'lon':LON.values })
print(np.shape(u_ano))
 
## ------------- b. 复杂网络
ds0=xr.open_dataset("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/Function/U300Positive_significant_995.nc")
 
# 调整经度坐标
for lon_name in ['lon1', 'lon2']:
    ds0['longitude_adjusted'] = xr.where(ds0[lon_name] < 0, ds0[lon_name] + 360, ds0[lon_name])
    ds0 = ds0.swap_dims({lon_name: 'longitude_adjusted'}).sel(
        **{'longitude_adjusted': sorted(ds0.longitude_adjusted)}
    ).drop(lon_name)
    ds0 = ds0.rename({'longitude_adjusted': lon_name})

lat=ds0.lat1; lon=ds0.lon1
network=ds0.significant_99
print(network)
 

ds1=xr.open_dataset("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/Function/U300Negative_significant_995.nc")
 
# 调整经度坐标
for lon_name in ['lon1', 'lon2']:
    ds1['longitude_adjusted'] = xr.where(ds1[lon_name] < 0, ds1[lon_name] + 360, ds1[lon_name])
    ds1 = ds1.swap_dims({lon_name: 'longitude_adjusted'}).sel(
        **{'longitude_adjusted': sorted(ds1.longitude_adjusted)}
    ).drop(lon_name)
    ds1 = ds1.rename({'longitude_adjusted': lon_name})

network_easterly=ds1.significant_99
 

# # ------------- c. 构建三极子指数


# triple_index = np.zeros((45,29+1,np.shape(u)[2]))

# for ilat in range(11,69+2,2):   #69
#     print(ilat)
    
#     for ilon in range(1,359+2,2):    # 359
        
#         # lon 要环绕处理   1, 3, 5, ..., 359  
#         # lon1_val = (ilon // 2) * 2 + 1

#         lat_range = np.arange(ilat-2,ilat+18+2)   # 不只是单数       # lat 要环绕处理    2 22
#         lat_range_val = [(x // 2) * 2 + 1 for x in lat_range]
#         lat2_range_val = list(set(lat_range_val)) 

#         lon_range = np.arange(ilon-40,ilon+40+2) %360  # 不只是单数       # lon 要环绕处理
#         lon_range_val = [(x // 2) * 2 + 1 for x in lon_range]
#         lon2_range_val = list(set(lon_range_val)) 
                                                            

#         # 1. 北边西风点
#         patch_north = network.sel(lat1=ilat, lon1=ilon, lat2=lat2_range_val, lon2 = lon2_range_val)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
#         north_best_lat, north_best_lon , north_max_val = find_max_patch(patch_north,lon1=ilon,lat_vals=lat2_range_val)
       

#         # 东风区：纬度 ilat-15 到 ilat+5
#         lat_range_east = np.arange(ilat - 16, ilat + 4+2)  # -14   6
#         lat_range_east_val = [(x // 2) * 2 + 1 for x in lat_range_east]
#         lat2_range_east_val = list(set(lat_range_east_val)) 

#         # 2. 中间东风
#         patch_center = network_easterly.sel(lat1=ilat, lon1=ilon, lat2=lat2_range_east_val, lon2 = lon2_range_val)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
#         center_best_lat, center_best_lon, center_max_val = find_max_patch(patch_center, lon1=ilon,lat_vals=lat2_range_east_val)


#         # 南边东风：纬度 ilat-30 到 ilat-15
#         lat_range_center = np.arange(ilat - 32, ilat - 12+2) #-28-8
#         lat_range_center_val = [(x // 2) * 2 + 1 for x in lat_range_center]
#         lat2_range_center_val = list(set(lat_range_center_val)) 

#         # 3. 南边西风
#         patch_south = network.sel(lat1=ilat, lon1=ilon, lat2=lat2_range_center_val, lon2 = lon2_range_val)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
#         south_best_lat, south_best_lon, south_max_val = find_max_patch(patch_south, lon1=ilon,lat_vals=lat2_range_center_val)

 
#         triple_index[:,(ilat-11)//2, (ilon-1)//2] =np.mean(u_ano.sel(lat=north_best_lat,lon=north_best_lon),axis=(1,2))\
#                     + np.mean(u_ano.sel(lat=south_best_lat,lon=south_best_lon),axis=(1,2))\
#                     - 2* np.mean(u_ano.sel(lat=center_best_lat,lon=center_best_lon),axis=(1,2))
       


# # triple_index_scale = standardization(triple_index)
# # print(triple_index_scale)


# ##--##--##-- 存储 --##--##--##
# tripole = xr.DataArray(data= triple_index, dims=['year','lat','lon'],
#             coords= {'year':range(45) , 'lat':range(11,69+2,2),'lon':range(1,359+2,2)})

# ds3 = xr.Dataset(data_vars= dict(tripole = tripole))
# ds3.to_netcdf("/public/home/fcai/abc/2Paper_blocking/Fig4/Dect_best_point/th995_tripole_wind_index_based_on_automatic_finding.nc")
# ds3.close()
# exit()
# del network

ds3=xr.open_dataset("/public/home/fcai/abc/2Paper_blocking/Fig4/Dect_best_point/th995_tripole_wind_index_based_on_automatic_finding.nc")
lat1=ds3.lat; lon1=ds3.lon
triple_index_scale=ds3.tripole 
triple_index_scale = standardization(triple_index_scale)
 
print(triple_index_scale)

## ------------- c. 计算趋势 
nt = 45;nlat=len(lat1);nlon= len(lon1)
U_result=trend_result(triple_index_scale,nt,nlat,nlon)
u_p_values_grd = U_result[1]
u_anom_grd = U_result[0]
u_anom_grd = xr.DataArray(data= u_anom_grd, dims=['lat','lon' ],
            coords= {'lat':range(11,69+2,2),'lon':range(1,359+2,2)})

detrended_triple_index_scale = signal.detrend(triple_index_scale, axis=0, type='linear', 
                                    bp=0, overwrite_data=False)


w = np.cos(np.deg2rad(triple_index_scale.lat))

obser_WNA = triple_index_scale.loc[:,40:52,235:250].weighted(w).mean(dim=("lat","lon"))
obser_EE  = triple_index_scale.loc[:,45:65,18:56].weighted(w).mean(dim=("lat","lon"))
obser_CA  = triple_index_scale.loc[:,33:48,88:106].weighted(w).mean(dim=("lat","lon"))
obser_NA  = triple_index_scale.loc[:,58:69,135:175].weighted(w).mean(dim=("lat","lon"))


nt=45
obser_WNA_trend = trend_1D_result(obser_WNA.values ,nt)
obser_EE_trend = trend_1D_result(obser_EE.values ,nt)
obser_CA_trend = trend_1D_result(obser_CA.values ,nt)
obser_NA_trend = trend_1D_result(obser_NA.values ,nt)
print("obser_dyn_WNA",obser_WNA_trend)
print("obser_dyn_CA",obser_CA_trend)
print("obser_dyn_EE",obser_EE_trend)
print("obser_dyn_NA",obser_NA_trend)
# ## ------------- 2. explained variance
from scipy.stats import pearsonr

##--##--##--##--  a. 输入Dyn 75th 数据  --##--##--##--##

ds1 = xr.open_dataset("/public/home/fcai/abc/2Paper_blocking/Fig4/T_dyn_therm_anom_19792023_JJA_NH_daily2x2.nc")
ds1.coords['lon'] = np.mod(ds1['lon'], 360)
ds1 = ds1.reindex({ 'lon' : np.sort(ds1['lon'])})
print(ds1)
 
T_dyn_anom = ds1.T_dyn_anom.loc[:,11:69,:] 
lat = ds1.lat.loc[11:69];   lon = ds1.lon 
year=T_dyn_anom.time.dt.year
month=T_dyn_anom.time.dt.month
 
## ------------- b. 读取 动态气温数据 (1979-2023, 7-8月)
year_ind=(year>=1979)&(year<=2023)

T_dyn_Jul=T_dyn_anom.where(year_ind&(month==7),drop=True)
T_dyn_Aug=T_dyn_anom.where(year_ind&(month==8),drop=True)

T_dyn_4D=np.zeros((45,62,30,nlon))

for iyear in range(1979,2024):
    i=iyear-1979
    T_dyn_4D[i,0:31,:,:]=T_dyn_Jul[i*31+0:i*31+31,:,:]
    T_dyn_4D[i,31:62,:,:]=T_dyn_Aug[i*31+0:i*31+31,:,:]
   

##--##--##--##--  计算每个夏天的 75 分位阈值  --##--##--##--##
t_th75 = np.percentile(T_dyn_4D, 75, axis=1)   ;print(np.shape(t_th75)) ## 计算75%分位数值
t_th75 = xr.DataArray(data= t_th75, dims=['year','lat','lon'],
              coords= {'year':np.linspace(1979,2023,45) ,'lat':lat.data , 'lon':lon1})


## ------------- d. 求相关系数
 
ngrd=nlat*nlon;nt=45
x = triple_index_scale.values.reshape((nt, ngrd), order='F')
y = t_th75.values.reshape((nt, ngrd), order='F')

# 建立斜率和显著性数组
correlation = np.empty((ngrd))
p_values = np.empty((ngrd))

# 计算每个网格点的线性趋势和显著性
for i in range(ngrd):
    r, p_value = pearsonr(x[:,i], y[:,i])
    correlation[i] = r
    p_values[i] = p_value


# 将斜率和显著性结果重新调整为原来的网格形状
day_shape = (nlat, nlon)
corre_anom_grd = correlation.reshape(day_shape, order='F')
corre_p_values_grd = p_values.reshape(day_shape, order='F')

 


## ------------- 3. Estimated trend
# t_th75 = np.where( t_th75==0.0,np.nan,t_th75   )
print(np.shape(t_th75))    ## 输出变量维度

t_th75 = xr.DataArray(data= t_th75, dims=['year','lat','lon'],
              coords= {'year':np.linspace(1979,2023,45) ,'lat':lat.data , 'lon':lon1.data})

##--##--## 计算趋势 --##--##--##
nt = 45
T_trend=trend_result(t_th75,nt,nlat,nlon)
t_p_values_grd = T_trend[1]
t_anom_grd = T_trend[0]

# 求回归系数
ngrd=nlat*nlon; nt=45
x = detrended_triple_index_scale.reshape((nt, ngrd), order='F')
y = t_th75.values.reshape((nt, ngrd), order='F')

# 建立斜率和显著性数组
regression = np.empty((ngrd))
p_values = np.empty((ngrd))


# 计算每个网格点的线性趋势和显著性
for i in range(ngrd):

    if np.all(x[:, i] == x[0, i]):
        print(f"Identical x values in column {i}")
    slope, _, _, p_value, _ = stats.linregress(x[:,i], y[:,i])
    regression[i] = slope
    p_values[i] = p_value


# 将斜率和显著性结果重新调整为原来的网格形状
day_shape = (nlat, nlon)
anom_grd = regression.reshape(day_shape, order='F')



## ------------- 5. 求Estimated trend
 
Estimated_trend= u_anom_grd *anom_grd 

Estimated_trend = xr.DataArray(data= Estimated_trend, dims=['lat','lon'],
              coords= {'lat':lat.data , 'lon':lon1.data})
# test = np.where(((Estimated_trend>0) & (t_anom_grd>0)) | ((Estimated_trend<0) & (t_anom_grd<0)),1,0 )
test_red = np.where((t_anom_grd>0) & (Estimated_trend> 0.3*t_anom_grd),1,0)
test_blue = np.where(((t_anom_grd<0) & (Estimated_trend< 0.3*t_anom_grd)),1,0 )



## -------------2. 指数趋势


## -------------a. historical model average change

 
ds3=xr.open_dataset("/public/home/fcai/abc/2Paper_blocking/Final_figures/Revision_Fig1/standize_th995_historical_Model_tripole_wind_index_7923.nc")
lat2=ds3.lat; lon2=ds3.lon
triple_index=ds3.tripole  
 
# 循环model name
historical_U_trend = np.zeros((20,29,180))
for model in range(20):

    print(model)
    
    triple_index[model,:,:,:] = standardization(triple_index[model,:,:,:])
    ##--##--## 计算趋势 --##--##--##
    nt = 45;nlat=29;nlon=180
    U_trend=trend_result(triple_index[model,:,:,:],nt,nlat,nlon)
    anom_grd = U_trend[0]
    historical_U_trend[model,:,:] = anom_grd

model_U_grd = np.mean(historical_U_trend,0)
historical_U_trend = xr.DataArray(data= historical_U_trend, dims=['model','lat','lon' ],
            coords= {'model':range(20),'lat':range(12,68+2,2),'lon':range(0,358+2,2)})
model_U_grd = xr.DataArray(data= model_U_grd, dims=['lat','lon' ],
            coords= {'lat':range(12,68+2,2),'lon':range(0,358+2,2)})


historical_U_trend_interp = historical_U_trend.interp(lat=u_anom_grd.lat, lon=u_anom_grd.lon, method='linear')

# Raw
w = np.cos(np.deg2rad(historical_U_trend.lat))

Model_WNA_trend = historical_U_trend.loc[:,40:52,235:250].weighted(w).mean(dim=("lat","lon")).mean(dim="model")
Model_EE_trend  = historical_U_trend.loc[:,45:65,18:56].weighted(w).mean(dim=("lat","lon")).mean(dim="model")
Model_CA_trend  = historical_U_trend.loc[:,33:48,88:106].weighted(w).mean(dim=("lat","lon")).mean(dim="model")
Model_NA_trend  = historical_U_trend.loc[:,58:69,135:175].weighted(w).mean(dim=("lat","lon")).mean(dim="model")
print("Model_dyn_WNA",Model_WNA_trend)
print("Model_dyn_CA",Model_CA_trend)
print("Model_dyn_EE",Model_EE_trend)
print("Model_dyn_NA",Model_NA_trend)

# -------------- 验证 
U_test_his =np.zeros((20,30,180))
U_test_his =np.where((np.tile(u_anom_grd,(20,1,1))>0)&(historical_U_trend_interp>0)|
                    (np.tile(u_anom_grd,(20,1,1))<0)&(historical_U_trend_interp<0), 1, 0)    

Model_U_test =  np.sum(U_test_his,0)  #20*90% = 18
Model_U_test = xr.DataArray(data= Model_U_test, dims=['lat','lon'],
              coords= {'lat':historical_U_trend_interp.lat.values,'lon':historical_U_trend_interp.lon.values})
print("Model_U_test",Model_U_test)


# -------------- 验证 --------------
# 审稿人的 一致
# -----------------------------------
# Model_sign = np.sign(historical_U_trend_interp)
# Obser_sign = np.sign(u_anom_grd)

# # 观测每个模式是否和观测一样
# Agreement = np.where(Model_sign== np.expand_dims(Obser_sign,axis=0), 1,0)
# Agree_num = np.sum(Agreement, axis=0)
# Agree_ratio=Agree_num/20.0


# from scipy.stats import binom
# from statsmodels.stats.multitest import fdrcorrection

# nmodel=20

# # 每个格点 得到 >=当前一致数量的概念; 每次成功概率0.5,看看他们的可能是不是随机的flunk
# p_value = binom.sf(Agree_num -1, nmodel,0.5)


# # FDR correction
# reject, p_fdr = fdrcorrection(p_value.flatten(), alpha=0.1)
# FDR_mask = reject.reshape(p_value.shape)

# # 最终 inter-model agreement & 模式和观测的一致性
# agreement_mask = (Agree_ratio>=0.7) & (FDR_mask)


 
#----------------------分割线（画图函数）------------------------
with open('/public/home/fcai/abc/data/CN-border-La.dat') as src:
    context = src.read()
    blocks = [cnt for cnt in context.split('>') if len(cnt) > 0]
    borders = [np.fromstring(block, dtype=float, sep=' ') for block in blocks]


# Create a figure and axis
# fig = plt.figure(figsize=[8, 12],frameon=True)
# ax1 = fig.add_axes([0, 0.75, 0.9, 0.7], projection=ccrs.PlateCarree(central_longitude=150) )
# ax2 = fig.add_axes([0.0, 0.5, 0.9, 0.7], projection=ccrs.PlateCarree(central_longitude=150) )
# ax3 = fig.add_axes([0.0, 0.25, 0.9, 0.7], projection=ccrs.PlateCarree(central_longitude=150) )
 
projection = ccrs.Robinson(central_longitude=150)

fig = plt.figure(figsize=[8, 12],frameon=True,dpi=300)
ax1 = fig.add_axes([0, 0.75, 0.9, 0.7], projection=projection )
ax2 = fig.add_axes([0.87, 0.75, 0.9, 0.7], projection=projection )
ax3 = fig.add_axes([0.0, 0.45, 0.9, 0.7], projection=projection )
ax4 = fig.add_axes([0.87, 0.45, 0.9, 0.7], projection=projection )
ax5 = fig.add_axes([1.75, 1.027, 0.13, 0.147] )
ax6 = fig.add_axes([1.75, 0.727, 0.13, 0.147] )

# ax22 = fig.add_axes([0.47, 0.72,0.17, 0.06])
 
# # prepare 
box = [-180, 180, 10, 70]  
scale = '50m'            
xstep, ystep = 90, 30

def make_map_Eurasia(ax):
    ax.set_extent(box, crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(box[0],box[1]+xstep, xstep),crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(box[2], box[3]+ystep, ystep),crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label =False))#经度0不加标识
    ax.tick_params(which='major', 
                    direction='out', 
                    length=4,
                    width=0.99, 
                    pad=3, 
                    labelsize=10,
                    bottom=True, left=True, right=False, top=False)
    ax.add_feature(cfeature.COASTLINE,  linewidth=0.4 , edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linestyle = 'dotted',lw = 0.35, edgecolor='black')
    return ax
 
#---------------------- 绘制趋势结果 ------------------------


cmap = plt.get_cmap('PuOr')
cmap_colors = cmap(np.arange(cmap.N))
cmap_reversed = mcolors.LinearSegmentedColormap.from_list('Reversed', cmap_colors[::-1])



#设置色标的
cbar_kwargs = {
    'orientation': 'horizontal',  # 改为水平方向
    'label': 'm/s/decade',
    'ticks': np.arange(-0.6,  0.6 + 0.3, 0.3),
    'pad': 0.03,
    'shrink': 0.55,
    'extend': 'both',
    'extendfrac': 0.1  
}


#设置画数据的精度，这里设置了从-60到60度，按1度画温度
levels = np.arange(-0.6,  0.6 + 0.1, 0.1)



U_mask = mask_land(u_anom_grd,'ocean','lon')
global_t_composite2, cycle_lon = add_cyclic_point(U_mask, coord=lon)   ## 给数据加上 360度经度 位置的数据

mappable = ax1.contourf(range(1,359+2+2,2),range(11,69+2,2),global_t_composite2,transform=ccrs.PlateCarree(),cmap=cmap_reversed,levels=levels,extend='both')

p_t_value = u_p_values_grd
c1p = ax1.contourf( range(1,359+2,2),range(11,69+2,2),p_t_value[:,:], levels =[0,0.1,1],hatches=[ 'XXX',None],colors="none", transform=ccrs.PlateCarree())


for collection in c1p.collections:
    collection.set_edgecolor('white')
for collection in c1p.collections:
    collection.set_linewidth(0)

 
new_position = fig.add_axes([0.25 , 0.97,0.35, 0.011]) 
cb= fig.colorbar(mappable, norm=norm,  cax=new_position, **cbar_kwargs)
cb.ax.xaxis.set_tick_params(labelsize=9)
ax1.set_title(' Trends of TWAI ',loc='center',color='black',fontsize=12,y=1.03)
ax1.text(0.06,1.1, 'a', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax1.transAxes)

   
ax1.set_xlabel(' ')
ax1.set_ylabel(' ')
ax1.set_aspect(1.35)


ax1.axis('off') 
ax1.add_feature(cfeature.COASTLINE,  linewidth=0.5 , edgecolor='black')
ax1.add_feature(cfeature.BORDERS, linestyle = 'dotted',lw = 0.35, edgecolor='black')
cb.outline.set_color('none')
cb.update_ticks()
cb.ax.xaxis.set_tick_params(labelsize=9)


# #---------------------- 2. 绘制趋势结果 ------------------------


#设置色标的
tbar_kwargs = {
    'orientation': 'horizontal',  # 改为水平方向
    'label': '%',
    'ticks': [0.1,0.2,0.3,0.4,0.5,0.6,0.7],
    'pad': 0.05,
    'shrink': 0.65,
    'extend': 'both',
    'extendfrac': 0.1  
}

#设置画数据的精度，这里设置了从-60到60度，按1度画温度
levels = np.arange(0.2,  0.7 + 0.1, 0.1)
Colors=('#fef5b2','#ffdf82','#fdb851','#fd8c3c','#fa4727','#d8141e','#a20027')


corre_anom_grd2 = xr.DataArray(data= corre_anom_grd*corre_anom_grd, dims=['lat','lon' ],
            coords= {'lat':lat1,'lon':lon1})


global_t_composite2, cycle_lon = add_cyclic_point(corre_anom_grd2, coord=lon)   ## 给数据加上 360度经度 位置的数据
mappable2 = ax2.contourf(range(1,359+2+2,2),lat1,global_t_composite2,transform=ccrs.PlateCarree(),colors=Colors,levels=levels,extend='both')

p_t_value = corre_p_values_grd
c2p = ax2.contourf( range(1,359+2,2),lat1,p_t_value[:,:], levels =[0,0.01,1],hatches=[ '///',None],linewidth=0.8,colors="none", transform=ccrs.PlateCarree())#alpha=0.3,

for collection in c2p.collections:
    collection.set_edgecolor('yellow')
    collection.set_linewidth(0)


# ax1.axis('off')    
new_position = fig.add_axes([1.2 , 0.97,0.35, 0.011]) 
cb= fig.colorbar(mappable2, norm=norm,  cax=new_position, **tbar_kwargs)

cb.outline.set_color('none')
cb.update_ticks()
cb.set_ticks([0.2,0.3,0.4,0.5,0.6,0.7])
cb.set_ticklabels([20,30,40,50,60,70])
cb.ax.xaxis.set_tick_params(labelsize=9)
ax2.set_title('Explained variance of TWAI to T$_{cir}$',loc='center',color='black',fontsize=12,y=1.03)
ax2.set_xlabel(' ')
ax2.set_ylabel(' ')
ax2.set_aspect(1.35)


ax2.axis('off') 
ax2.add_feature(cfeature.COASTLINE,  linewidth=0.5 , edgecolor='black')
ax2.add_feature(cfeature.BORDERS, linestyle = 'dotted',lw = 0.35, edgecolor='black')
ax2.text(0.06,1.1,'c', fontsize=15.5,fontweight='bold', fontfamily='sans-serif',transform=ax2.transAxes)



#---------------------- 3.  ------------------------
#设置色标的
u_cbar_kwargs = {
    'orientation': 'horizontal',  # 改为水平方向
    'label': 'm/s/decade',
    'ticks': np.arange(-0.6,  0.6 + 0.3, 0.3),
    'pad': 0.03,
    'shrink': 0.55,
    'extend': 'both',
    'extendfrac': 0.1  
}


#设置画数据的精度，这里设置了从-60到60度，按1度画温度
levels = np.arange(-0.6,  0.6 + 0.1, 0.1)


#设置画数据的精度，这里设置了从-60到60度，按1度画温度
levels = np.arange(-0.6,  0.6 + 0.1, 0.1) # np.arange(-0.3,  0.3 + 0.05, 0.05)
Colors=('#2c0248',"#450e69",'#5f3a91','#8477af','#ada6ce','#cfcfe5','#ffffff','#ffffff','#fde1b5','#fab35b','#e2861a','#bd6109',"#985524",'#75370c')
cmap = plt.get_cmap('PuOr')
cmap_colors = cmap(np.arange(cmap.N))
cmap_reversed = mcolors.LinearSegmentedColormap.from_list('Reversed', cmap_colors[::-1])

model_U_grd = mask_land(model_U_grd,'ocean','lon') 

global_t_composite2, cycle_lon = add_cyclic_point(model_U_grd, coord=lon1)   ## 给数据加上 360度经度 位置的数据
mappable = ax3.contourf(range(0,358+2+2,2),range(12,68+2,2),global_t_composite2,transform=ccrs.PlateCarree(),cmap=cmap_reversed,levels=levels,extend='both')

# agreement_plot = np.where(Model_U_test, 1, np.nan)
c1p = ax3.contourf( Model_U_test.lon,Model_U_test.lat, Model_U_test, levels =[0,13.5,20],hatches=[None, 'XXX'],colors="none", transform=ccrs.PlateCarree())


for collection in c1p.collections:
    collection.set_edgecolor('white')
for collection in c1p.collections:
    collection.set_linewidth(0)

 
new_position = fig.add_axes([0.25, 0.67,0.35, 0.011 ]) 
cbar=fig.colorbar(mappable,cax=new_position,**u_cbar_kwargs)
cbar.outline.set_color('none')
cbar.update_ticks()
cbar.ax.xaxis.set_tick_params(labelsize=10)

ax3.set_title(' Multi-model mean of trends in TWAI',loc='center',color='black',fontsize=13,y=1.03)
ax3.text(0.06,1.1, 'b', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax3.transAxes)
ax3.text(0.48,0.21, 'cross: ', fontsize=10.5,  fontfamily='sans-serif',transform=ax3.transAxes)
ax3.text(0.48,0.08, '≥14 models = ERA5 sign', fontsize=10,  fontfamily='sans-serif',transform=ax3.transAxes)

ax3.set_xlabel(' ')
ax3.set_ylabel(' ')
ax3.set_aspect(1.5)


ax3.axis('off') 
ax3.add_feature(cfeature.COASTLINE,  linewidth=0.5 , edgecolor='black')
ax3.add_feature(cfeature.BORDERS, linestyle = 'dotted',lw = 0.35, edgecolor='black')



# ##### 小图
# shops = [ "EE", "NWC", "ES","WNA"]
# MODEL = [Model_EE_trend, Model_CA_trend, Model_NA_trend,Model_WNA_trend ]
# OBSER = [obser_EE_trend,obser_CA_trend,obser_NA_trend,obser_WNA_trend]

# # 第二根柱子“堆积”在第一根柱子上方，通过'bottom'调整，显示第二种产品的销量
# ax22.bar(shops, OBSER, color="#ec6e00", label="Obser",width=0.35)#bottom=sales_product_1
# # 先创建一根柱子，显示第一种产品的销量
# ax22.bar(shops, MODEL, color="#f9bc4a", label="Model",width=0.27)



# ax22.legend(frameon=False, loc='upper right',fontsize=8,bbox_to_anchor=(0.95,1.58))
# ax22.spines['right'].set_visible(False)
# ax22.spines['top'].set_visible(False)
# ax22.set_title(" ", loc='left', color='black', fontsize=10)
# ax22.set_facecolor('none')
# ax22.set_ylim(0,0.2)
# ax22.tick_params(axis='both',labelsize=8, length = 3, width=0.49)
# # for artist in ax22.get_children():
# #     artist.set_alpha(0.3)

#---------------------- 4. 绘制趋势结果 ------------------------


#设置色标的
cbar_kwargs = {
    'orientation': 'horizontal',  # 改为水平方向
    'label': '°C/decade',
    'ticks': np.arange(-0.4, 0.4 + 0.2, 0.2),
    'pad': 0.03,
    'shrink': 0.55,
    'extend': 'both',
    'extendfrac': 0.1  
}

#设置画数据的精度，这里设置了从-30到30度，按1度画温度
levels = np.arange(-0.4, 0.4 + 0.05, 0.05)

# 创建一个自定义的颜色映射（例如：'RdYlGn'）
cmap = plt.get_cmap('RdBu')
cmap_colors = cmap(np.arange(cmap.N))
cmap_reversed = mcolors.LinearSegmentedColormap.from_list('Reversed', cmap_colors[::-1])


Estimated_trend= mask_land(Estimated_trend,'ocean','lon'); print(Estimated_trend.values)

Estimated_trend_grd = xr.DataArray(data= Estimated_trend, dims=['lat','lon' ],
            coords= {'lat':lat1,'lon':lon1})


global_t_composite3, cycle_lon = add_cyclic_point(Estimated_trend_grd, coord=lon)   ## 给数据加上 360度经度 位置的数据
mappable3 = ax4.contourf(range(1,359+2+2,2),lat1,global_t_composite3,transform=ccrs.PlateCarree(),cmap=cmap_reversed,levels=levels,extend='both')
 
c3p = ax4.contourf( range(1,359+2,2),lat1,test_red, levels =[0,0.5,1],hatches=[None, '///'],linewidth=0.8, colors="none", transform=ccrs.PlateCarree())
for collection in c3p.collections:
    collection.set_edgecolor("#d86976")
    collection.set_linewidth(0)

c4p = ax4.contourf( range(1,359+2,2),lat1,test_blue, levels =[0,0.5,1],hatches=[None, '///'],linewidth=0.5,colors="none", transform=ccrs.PlateCarree())
for collection in c4p.collections:
    collection.set_edgecolor("#275c8b")
    collection.set_linewidth(0)

# ax1.axis('off')    
new_position = fig.add_axes([1.2, 0.67,0.35, 0.011]) 
cb= plt.colorbar(mappable3,cmap=cmap_reversed,norm=norm,  cax=new_position,**cbar_kwargs)
# cbar.outline.set_color('none')
# cbar.update_ticks()
cb.ax.xaxis.set_tick_params(labelsize=9)
ax4.set_title(' Estimated trends of T$_{cir}$',loc='center',color='black',fontsize=12,y=1.03)
ax4.set_xlabel(' ')
ax4.set_ylabel(' ')
ax4.set_aspect(1.35)



ax4.axis('off') 
ax4.add_feature(cfeature.COASTLINE,  linewidth=0.5 , edgecolor='black')
ax4.add_feature(cfeature.BORDERS, linestyle = 'dotted',lw = 0.35, edgecolor='black')
cb.outline.set_color('none')
cb.update_ticks()
cb.ax.xaxis.set_tick_params(labelsize=9)
ax4.text(0.06,1.1,'d', fontsize=15.5,fontweight='bold', fontfamily='sans-serif',transform=ax4.transAxes)



# Raw
w = np.cos(np.deg2rad(Estimated_trend.lat))

T_WNA_trend = Estimated_trend.loc[40:52,235:250].weighted(w).mean(dim=("lat","lon"))
T_EE_trend  = Estimated_trend.loc[45:65,22:60].weighted(w).mean(dim=("lat","lon"))
T_CA_trend  = Estimated_trend.loc[30:45,88:106].weighted(w).mean(dim=("lat","lon"))
T_NA_trend  = Estimated_trend.loc[58:69,135:175].weighted(w).mean(dim=("lat","lon"))


print("      Estimate_T_WNA",T_WNA_trend)# ,"      rate",(dyn_WNA_trend/T_WNA_trend ))
print("      Estimate_T_EE",T_EE_trend)#,"      rate",str(dyn_EE_trend/T_EE_trend ))
print("      Estimate_T_CA",T_CA_trend)#,"      rate",str(dyn_CA_trend/T_CA_trend ))
print("      Estimate_T_NA",T_NA_trend)#,"      rate",str(dyn_NA_trend/T_NA_trend ))


ax4.text(0.61,0.6, f'{T_WNA_trend:0.2f}/0.17', fontsize=10, transform=ax4.transAxes)

ax4.text(0.09,0.85, f'{T_EE_trend:0.2f}/0.26', fontsize=10, transform=ax4.transAxes)
ax4.text(0.43,0.315, f'{T_CA_trend:0.2f}/0.24', fontsize=10, transform=ax4.transAxes)




#---------------------- 5. 解释百分比 ------------------------
lat_R2= np.nanmean(corre_anom_grd2 ,1)
x = lat1.values
y= lat_R2
 
ax5.plot( y,x, color='#ff0000', linewidth=1.5 , linestyle='-')
ax5.tick_params(axis='both',labelsize=10, length = 3, width=0.49)

ax5.set_yticks([20,40,60],['20N','40N','60N'],fontsize=10)
ax5.set_yticks([10,20,30,40,50,60,70],minor=True)
ax5.set_ylim(10,70)
ax5.set_xticks([0.2,0.3,0.4,0.5],['20','30','40 ','50'],fontsize=10)

ax5.yaxis.tick_right()
ax5.text(-0.2,1.06,'e', fontsize=15.5,fontweight='bold', fontfamily='sans-serif',transform=ax5.transAxes)
ax5.text(1.1,-0.104,'%', fontsize=10,transform=ax5.transAxes)
 

# 自定义Y轴次要刻度的外观
ax5.tick_params(axis='y', which='minor', left=False, right=True, length=4, width=1)
ax5.axvline(0.4,color='#ffa500', linestyle='--',linewidth=1.5)
ax5.set_title(' R\u00b2 (lat)',loc='center',color='black',fontsize=12,y=1.03)









#---------------------- 6. 绘制趋势结果 ------------------------
 
lat_grid= np.where(np.isnan(Estimated_trend),0,1)
syn_signal_grid = np.where((t_anom_grd>0) & (Estimated_trend_grd> 0) |((t_anom_grd<0) & (Estimated_trend_grd< 0)),1,0)
syn_signal_percent = np.sum(syn_signal_grid,1) / np.sum(lat_grid,1)
# print(syn_signal_percent)
# print(np.sum(syn_signal_grid,1))
# print(np.sum(lat_grid,1))
# trend_percent = xr.DataArray(data= trend_percent, dims=['lat','lon'],
#               coords= {'lat':lat.data , 'lon':lon1.data})


ax6.plot( syn_signal_percent,lat1.values, color='#ff0000', linewidth=1.5 , linestyle='-')
ax6.tick_params(axis='both',labelsize=10, length = 3, width=0.49)

ax6.set_title(' same sign',loc='center',color='black',fontsize=12,y=1.03)
ax6.set_xlim(0.40,0.9);ax6.set_ylim(10,70)
ax6.set_xticks([0.50,0.7,0.9],['50','70','90'],fontsize=10)
ax6.set_yticks([20,40,60],['20N','40N','60N'],fontsize=10)
ax6.set_xticks([0.4,0.5,0.6,0.7,0.8,0.9],minor=True)
ax6.set_yticks([10,20,30,40,50,60,70],minor=True)



ax6.yaxis.tick_right()
ax6.text(-0.2,1.06,'f', fontsize=15.5,fontweight='bold', fontfamily='sans-serif',transform=ax6.transAxes)
ax6.text(1.1,-0.114,'%', fontsize=10,transform=ax6.transAxes)
 

# 自定义Y轴次要刻度的外观
ax6.tick_params(axis='y', which='minor', left=False, right=True, length=4, width=1)
ax6.axvline(0.7,color='#ffa500', linestyle='--',linewidth=1.5)








# 绘制一个矩形框(NE US)


for k in range(1,5):


    # West NA
    locals()['rectangle' + str(k)] = Rectangle(
        (235, 40),  # 左下角坐标 (longitude, latitude)
        15, 12, # 宽度和高度 (longitude, latitude)
        linewidth=1.5,edgecolor='black',
        facecolor='none',linestyle='--',
        transform=ccrs.PlateCarree()
    )
    locals()['ax' + str(k)].add_patch(locals()['rectangle' + str(k)]) 


    #East Europe
    locals()['rectangle' + str(k)] = Rectangle(
        (18, 45),
        38, 20, 
        linewidth=1.3,edgecolor='black',
        facecolor='none',linestyle='--',
        transform=ccrs.PlateCarree()
    )
    locals()['ax' + str(k)].add_patch(locals()['rectangle' + str(k)])


    #Central Asia
    locals()['rectangle' + str(k)] = Rectangle(
        (88, 33),  # 左下角坐标 (longitude, latitude)
        18, 15, # 宽度和高度 (longitude, latitude)
        linewidth=1.3,edgecolor='black',
        facecolor='none',linestyle='--',
        transform=ccrs.PlateCarree()
    )
    locals()['ax' + str(k)].add_patch(locals()['rectangle' + str(k)])  

    #North Asia
    locals()['rectangle' + str(k)] = Rectangle(
        (135, 58), 
        40, 11, # 宽度和高度 (longitude, latitude)
        linewidth=1.3,edgecolor='black',
        facecolor='none',linestyle='--',
        transform=ccrs.PlateCarree()
    )
    locals()['ax' + str(k)].add_patch(locals()['rectangle' + str(k)]) 


plt.savefig("/public/home/fcai/abc/2Paper_blocking/Final_figures/Figure3.pdf", dpi=500, bbox_inches='tight')
plt.show()
 