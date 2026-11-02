from cartopy.util import add_cyclic_point
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from copy import copy
from scipy import signal
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
import os
import glob
from global_land_mask import globe
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
#-*- coding:utf-8 –*-
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


## -------------1. Temperature 趋势

## -------------a. historical 

His = xr.open_dataset("/public/home/fcai/abc/2Paper_blocking/CMIP6_data/complex_network/6_pattern/data/2x2_Dynamical_Tmean_model_day_JA_19792023.nc")

His.coords['lon'] = np.mod(His['lon'], 360)
His = His.reindex({ 'lon' : np.sort(His['lon'])})

his_air = His.air_75.loc[:,:,12:68,:]
his_air = his_air.sel(year=(His.year>=1979)&(His.year<=2023));print(his_air)
his_th75 =xr.DataArray(data=his_air, dims=['models','year','lat','lon'],
                coords={'models':["ACCESS-CM2","CanESM5","EC-Earth3","EC-Earth3-Veg","FGOALS-g3","GFDL-CM4","INM-CM4-8","INM-CM5-0","IPSL-CM6A-LR",\
                "MIROC6","MPI-ESM1-2-HR","MPI-ESM1-2-LR","MRI-ESM2-0","NorESM2-LM",\
                "NorESM2-MM","TaiESM1","HadGEM3-GC31-LL", "HadGEM3-GC31-MM","KACE-1-0-G","UKESM1-0-LL"],'year':np.arange(1979,2023+1,1),'lat':His.lat.loc[12:68],'lon':His.lon})      # .sel(year=(His.year>=1979))
lat = His.lat.loc[12:68];print(His)
lon = His.lon

# 循环model name
historical_T_trend = np.zeros((20,29,180))
model_name=("ACCESS-CM2","CanESM5","EC-Earth3","EC-Earth3-Veg",
            "FGOALS-g3","GFDL-CM4","HadGEM3-GC31-LL",\
            "HadGEM3-GC31-MM","INM-CM4-8","INM-CM5-0","IPSL-CM6A-LR",\
            "KACE-1-0-G","MIROC6","MPI-ESM1-2-HR","MPI-ESM1-2-LR",\
            "MRI-ESM2-0","NorESM2-LM","NorESM2-MM","TaiESM1","UKESM1-0-LL") #20

imodel=0
for model in model_name:

    print(model,imodel)
    ##--##--## 计算趋势 --##--##--##
    nt = 45;nlat=29;nlon=180

    T_trend=trend_result(his_th75.sel(models=model),nt,nlat,nlon)
    anom_grd = T_trend[0]
    historical_T_trend[imodel,:,:] = anom_grd
    imodel+=1


historical_T_trend = xr.DataArray(
    data=historical_T_trend,
    dims=["models", "lat", "lon"],
    coords={ "models":["ACCESS-CM2","CanESM5","EC-Earth3","EC-Earth3-Veg",
            "FGOALS-g3","GFDL-CM4","HadGEM3-GC31-LL",\
            "HadGEM3-GC31-MM","INM-CM4-8","INM-CM5-0","IPSL-CM6A-LR",\
            "KACE-1-0-G","MIROC6","MPI-ESM1-2-HR","MPI-ESM1-2-LR",\
            "MRI-ESM2-0","NorESM2-LM","NorESM2-MM","TaiESM1","UKESM1-0-LL"],"lat": lat.values , "lon": lon.values}
)

 
 
# -------------b. future

Fut = xr.open_dataset("/public/home/fcai/abc/2Paper_blocking/CMIP6_data/complex_network/6_pattern/data/2x2_Dynamical_Tmean_model_day_JA_20152099.nc")

Fut.coords['lon'] = np.mod(Fut['lon'], 360)
Fut = Fut.reindex({ 'lon' : np.sort(Fut['lon'])})

fut_air = Fut.air_75.loc[:,:,12:68,:]
Fut_th75 = fut_air.sel(year=(Fut.year>=2055)&(Fut.year<=2099));print(Fut_th75)
Fut_th75 =xr.DataArray(data=Fut_th75, dims=['models','year','lat','lon'],
                coords={'models':["ACCESS-CM2","CanESM5","EC-Earth3","EC-Earth3-Veg","FGOALS-g3","GFDL-CM4","INM-CM4-8","INM-CM5-0","IPSL-CM6A-LR",\
                "MIROC6","MPI-ESM1-2-HR","MPI-ESM1-2-LR","MRI-ESM2-0","NorESM2-LM",\
                "NorESM2-MM","TaiESM1","HadGEM3-GC31-LL", "HadGEM3-GC31-MM","KACE-1-0-G","UKESM1-0-LL"],'year':np.arange(2055,2099+1,1),'lat':His.lat.loc[12:68],'lon':His.lon})      # .sel(year=(His.year>=1979))


# 循环model name
future_T_trend = np.zeros((20,29,180))

imodel=0
for model in model_name:

    print(model,imodel)
    ##--##--## 计算趋势 --##--##--##
    nt = 45;nlat=29;nlon=180
    T_trend=trend_result(Fut_th75.sel(models=model),nt,nlat,nlon)
    anom_grd = T_trend[0]
    future_T_trend[imodel,:,:] = anom_grd
    imodel+=1

future_T_trend = xr.DataArray(
    data=future_T_trend,
    dims=["models", "lat", "lon"],
    coords={ "models":["ACCESS-CM2","CanESM5","EC-Earth3","EC-Earth3-Veg",
            "FGOALS-g3","GFDL-CM4","HadGEM3-GC31-LL",\
            "HadGEM3-GC31-MM","INM-CM4-8","INM-CM5-0","IPSL-CM6A-LR",\
            "KACE-1-0-G","MIROC6","MPI-ESM1-2-HR","MPI-ESM1-2-LR",\
            "MRI-ESM2-0","NorESM2-LM","NorESM2-MM","TaiESM1","UKESM1-0-LL"],"lat": lat.values , "lon": lon.values}
)

fut_model_anom_grd = np.mean(future_T_trend,0)
fut_model_anom_grd = xr.DataArray(
    data=fut_model_anom_grd,
    dims=[ "lat", "lon"],
    coords={ "lat": lat.values , "lon": lon.values}
)

fut_model_anom_grd = mask_land(fut_model_anom_grd,'ocean','lon') 




# -------------2. 指数趋势

#   指数按  historical 的没有去趋势， 但是future 的去了

# -------------a. historical model average change


ds3=xr.open_dataset("/public/home/fcai/abc/2Paper_blocking/Final_figures/Revision_Fig1/standize_th995_historical_Model_tripole_wind_index_7923.nc")
lat1=ds3.lat; lon1=ds3.lon
triple_index_scale=ds3.tripole ;print(triple_index_scale)

# 循环model name
historical_U_trend = np.zeros((20,29,180))
for model in range(20):
    # triple_index_scale[model,:,:,:] = standardization(triple_index_scale[model,:,:,:])

    print(model)
    ##--##--## 计算趋势 --##--##--##
    nt = 45;nlat=29;nlon=180
    U_trend=trend_result(triple_index_scale[model,:,:,:],nt,nlat,nlon)
    anom_grd = U_trend[0]
    historical_U_trend[model,:,:] = anom_grd

historical_U_trend = xr.DataArray(
    data=historical_U_trend,
    dims=["models", "lat", "lon"],
    coords={ "models":historical_T_trend.models,"lat": lat1.values , "lon": lon1.values}
)
 


## -------------b. future model average change


ds5=xr.open_dataset("/public/home/fcai/abc/2Paper_blocking/Final_figures/Revision_Fig1/standize_th995_future_Model_tripole_wind_index_based_on_automatic_finding.nc")
triple_index=ds5.tripole;print(triple_index) ;print(np.shape(triple_index))
lat1=ds5.lat; lon1=ds5.lon;print()
 

# 循环model name
future_U_trend = np.zeros((20,29,180))
for model in range(20):
    print(model)
    # triple_index[model,:,:,:] = standardization(triple_index[model,:,:,:])
    
    ##--##--## 计算趋势 --##--##--##
    nt = 45;nlat=29;nlon=180
    U_trend=trend_result(triple_index[model,:,:,:],nt,nlat,nlon)
    future_U_trend[model,:,:] = U_trend[0]


future_U_trend = xr.DataArray(data= future_U_trend, dims=['models','lat','lon'],
         coords= {'models':historical_U_trend.models,'lat':lat1.data , 'lon':lon1.data})

future_model_U = np.mean(future_U_trend,0)

# -------------- 验证 
U_test_fut =np.zeros((20,29,180))
U_test_fut = np.where( ((future_U_trend.values>0) & (future_model_U.values[None, :, :] > 0)) |
    ((future_U_trend.values < 0) & (future_model_U.values[None, :, :] < 0)), 1,0)

fut_Model_U_test = np.nansum(U_test_fut,0)  #20*90% = 18
fut_Model_U_test = xr.DataArray(data= fut_Model_U_test, dims=['lat','lon'],
         coords= {'lat':lat.data , 'lon':lon.data})



# # -------------- 验证 --------------
# # 每个模式和多模式平均的符号
# model_sign = np.sign(future_U_trend)
# ensemble_sign = np.sign(future_model_U)

# # xarray 会自动把 (lat, lon) 广播至 (models, lat, lon)
# agreement = xr.where(model_sign == ensemble_sign, 1, 0)

# agree_num = agreement.sum(dim="models")
# Agree_ratio=agree_num/20.0


# from scipy.stats import binom
# from statsmodels.stats.multitest import fdrcorrection

# nmodel=20

# # 每个格点 得到 >=当前一致数量的概念; 每次成功概率0.5,看看他们的可能是不是随机的flunk
# p_value = binom.sf(agree_num -1, nmodel,0.5)


# # FDR correction
# reject, p_fdr = fdrcorrection(p_value.flatten(), alpha=0.1)
# FDR_mask = reject.reshape(p_value.shape)

# # 最终 inter-model agreement & 模式和观测的一致性
# agreement_mask = (Agree_ratio>=0.7) & (FDR_mask)
# agreement_mask = xr.DataArray(data=agreement_mask, dims=['lat','lon'],
#          coords= {'lat':lat1.data , 'lon':lon1.values})

 



# ---------------------------- 3. Estimated trend


# ------------- a. historical model average change

# 求回归系数
nlat=29; nlon=180
ngrd=nlat*nlon; nt=45
# 循环model name
historical_Estimate = np.zeros((20,29,180))

imodel=0
for model in model_name:

    print(model,imodel)
    detrended_triple_index_scale = signal.detrend(triple_index_scale[imodel,:,:,:], axis=0, type='linear', 
                                    bp=0, overwrite_data=False)

    x = detrended_triple_index_scale.reshape((nt, ngrd), order='F')
    y = his_th75.sel(models=model).values.reshape((nt, ngrd), order='F')

    # 建立斜率和显著性数组
    regression = np.empty((ngrd))
    p_values = np.empty((ngrd))

    # 计算每个网格点的线性趋势和显著性
    for i in range(ngrd):

        slope, _, _, p_value, _ = stats.linregress(x[:,i], y[:,i])
        regression[i] = slope
        p_values[i] = p_value

    # 将斜率和显著性结果重新调整为原来的网格形状
    nlat=29; nlon=180
    day_shape = (nlat, nlon)
    anom_grd1 = regression.reshape(day_shape, order='F')
    ## ------------- 5. 求Estimated trend
 
    historical_Estimate[imodel,:,:]= historical_U_trend[imodel,:,:] *anom_grd1 
    imodel+=1

historical_Estimate = xr.DataArray(data= historical_Estimate, dims=['models','lat','lon'],
         coords= {'models':historical_T_trend.models,'lat':lat1.data , 'lon':lon1.data})



# ------------- b. future model average change

# 求回归系数
ngrd=29*180; nt=45

# 循环model name
future_Estimate = np.zeros((20,29,180))
imodel=0
for model in model_name:

    print(model,imodel)
    detrended_triple_index = signal.detrend(triple_index[imodel,:,:,:], axis=0, type='linear', 
                                    bp=0, overwrite_data=False) 

    X = detrended_triple_index.reshape((nt, ngrd), order='F')
    Y = Fut_th75.sel(models=model).values.reshape((nt, ngrd), order='F')

    # 建立斜率和显著性数组
    regression = np.empty((ngrd))
    p_values = np.empty((ngrd))

    # 计算每个网格点的线性趋势和显著性
    for i in range(ngrd):
        slope, _, _, p_value, _ = stats.linregress(X[:,i], Y[:,i])
        regression[i] = slope
        p_values[i] = p_value

    # 将斜率和显著性结果重新调整为原来的网格形状
    day_shape = (nlat, nlon)
    anom_grd2 = regression.reshape(day_shape, order='F')
    ## ------------- 5. 求Estimated trend
    
    future_Estimate[imodel,:,:]= future_U_trend[imodel,:,:]*anom_grd2 
    imodel+=1

future_Estimate = xr.DataArray(data= future_Estimate, dims=['models','lat','lon'],
         coords= {'models':future_T_trend.models,'lat':lat.values , 'lon':lon.values})


# 不同地方的方差
# Historical
w_his = np.cos(np.deg2rad(historical_Estimate.lat))

his_WNA_trend = historical_Estimate.loc[:,40:52,235:250].weighted(w_his).mean(dim=("lat","lon"))
his_EE_trend  = historical_Estimate.loc[:,45:65,18:56].weighted(w_his).mean(dim=("lat","lon"))
his_CA_trend  = historical_Estimate.loc[:,30:45,88:106].weighted(w_his).mean(dim=("lat","lon"))
his_NA_trend  = historical_Estimate.loc[:,58:69,135:175].weighted(w_his).mean(dim=("lat","lon"))


# Future
w_fut = np.cos(np.deg2rad(future_Estimate.lat))

fut_WNA_trend = future_Estimate.loc[:,40:52,235:250].weighted(w_fut).mean(dim=("lat","lon"))
fut_EE_trend  = future_Estimate.loc[:,45:65,18:56].weighted(w_fut).mean(dim=("lat","lon"))
fut_CA_trend  = future_Estimate.loc[:,30:45,88:106].weighted(w_fut).mean(dim=("lat","lon"))
fut_NA_trend  = future_Estimate.loc[:,58:69,135:175].weighted(w_fut).mean(dim=("lat","lon"))

print("      his_Estimate_T_WNA",np.mean(his_WNA_trend), "      fut",np.mean(fut_WNA_trend ))
print("      his_Estimate_T_EE",np.mean(his_EE_trend) ,"      fut",np.mean(fut_EE_trend ))
print("      his_Estimate_T_CA",np.mean(his_CA_trend ),"      fut",np.mean(fut_CA_trend ))
print("      his_Estimate_T_NA",np.mean(his_NA_trend) ,"      fut",np.mean(fut_NA_trend ))

print("      his_Estimate_T_WNA",his_WNA_trend, "      fut",fut_WNA_trend )
print("      his_Estimate_T_EE",his_EE_trend ,"      fut",fut_EE_trend )
print("      his_Estimate_T_CA",his_CA_trend ,"      fut",fut_CA_trend )
print("      his_Estimate_T_NA",his_NA_trend ,"      fut",fut_NA_trend )
# print( "      fut",np.mean(fut_WNA_trend ))
# print("      fut",np.mean(fut_EE_trend ))
# print("      fut",np.mean(fut_CA_trend ))
# print("      fut",np.mean(fut_NA_trend ))

 

#----------------------分割线（画图函数）------------------------
with open('/public/home/fcai/abc/data/CN-border-La.dat') as src:
    context = src.read()
    blocks = [cnt for cnt in context.split('>') if len(cnt) > 0]
    borders = [np.fromstring(block, dtype=float, sep=' ') for block in blocks]


projection = ccrs.Robinson(central_longitude=150) 

fig = plt.figure(figsize=[8, 12],frameon=True,dpi=300)
ax1 = fig.add_axes([0, 0.7, 0.7, 0.5], projection=projection )
ax2 = fig.add_axes([0.75, 0.7, 0.7, 0.5], projection=projection )
ax5 = fig.add_axes([0.1, 0.53, 0.5, 0.2] )
ax6 = fig.add_axes([0.85, 0.43, 0.5, 0.3] )


#---------------------- 绘制趋势结果 ------------------------


#设置色标的
cbar_kwargs = {
    'orientation': 'horizontal',  # 改为水平方向
    'label': '°C/decade',
    'ticks': np.arange(-0.3, 0.3 + 0.1, 0.1),
    'pad': 0.03,
    'shrink': 0.55,
    'extend': 'both',
    'extendfrac': 0.1  
}

#设置画数据的精度，这里设置了从-30到30度，按1度画温度
levels = np.arange(-0.3, 0.3 + 0.02, 0.02)

# 创建一个自定义的颜色映射（例如：'RdYlGn'）
cmap = plt.get_cmap('RdBu')
cmap_colors = cmap(np.arange(cmap.N))
cmap_reversed = mcolors.LinearSegmentedColormap.from_list('Reversed', cmap_colors[::-1])


# Setting Fig. 2

#设置画数据的精度，这里设置了从-60到60度，按1度画温度
levels = np.arange(-0.3,  0.3 + 0.05, 0.05)

global_t_composite3, cycle_lon3 = add_cyclic_point(fut_model_anom_grd, coord=lon1)   ## 给数据加上 360度经度 位置的数据
mappable3 = ax1.contourf(range(0,358+2+2,2),range(12,68+2,2),global_t_composite3,transform=ccrs.PlateCarree(),cmap=cmap_reversed,levels=levels,extend='both')


# -------------- 验证 
T_test_fut =np.zeros((20,29,180))
T_test_fut =np.where((future_T_trend>0)&(np.tile(fut_model_anom_grd,(20,1,1)) >0)|
                      (future_T_trend<0)&(np.tile(fut_model_anom_grd,(20,1,1)) <0)  , 1, 0)    

fut_Model_T_test = np.nansum(T_test_fut,0)  #20*90% = 18
fut_Model_T_test = xr.DataArray(data= fut_Model_T_test, dims=['lat','lon'],
         coords= {'lat':lat.data , 'lon':lon.values})
fut_Model_T_test = mask_land(fut_Model_T_test,'ocean','lon') 


# # -------------- 验证 --------------
# # 审稿人的 一致
# # -----------------------------------
# Model_sign = np.sign(fut_model_anom_grd)
# ensemble_sign = np.sign(future_T_trend)

# # 观测每个模式是否和观测一样
# Agreement = np.where(Model_sign== ensemble_sign, 1,0)
# Agree_num = np.sum(Agreement, axis=-1)
# Agree_ratio_T=Agree_num/20.0


# from scipy.stats import binom
# from statsmodels.stats.multitest import fdrcorrection

# nmodel=20

# # 每个格点 得到 >=当前一致数量的概念; 每次成功概率0.5,看看他们的可能是不是随机的flunk
# p_value = binom.sf(Agree_num -1, nmodel,0.5)


# # FDR correction
# reject, p_fdr = fdrcorrection(p_value.flatten(), alpha=0.1)
# FDR_mask = reject.reshape(p_value.shape)

# # 最终 inter-model agreement & 模式和观测的一致性
# agreement_mask_T = (Agree_ratio_T>=0.7) & (FDR_mask)
# print(np.shape(agreement_mask_T))
# agreement_mask_T = xr.DataArray(data=agreement_mask_T, dims=['lat','lon'],
#          coords= {'lat':lat1.data , 'lon':lon1.values})

c1p = ax1.contourf(range(0,358+2,2),range(12,68+2,2),fut_Model_T_test.loc[12:68,:] , levels = [0,13.5,20],hatches=[ None,'XXX'],colors="none", transform=ccrs.PlateCarree())

for collection in c1p.collections:
    collection.set_edgecolor('white')
for collection in c1p.collections:
    collection.set_linewidth(0)
ax1.axis('off') 

new_position = fig.add_axes([0.14, 0.85,0.35, 0.011])   
ax1.add_feature(cfeature.COASTLINE,  linewidth=0.5 , edgecolor='black')
ax1.add_feature(cfeature.BORDERS, linestyle = 'dotted',lw = 0.35, edgecolor='black')

cbar=fig.colorbar(mappable3,norm=norm,cax=new_position,**cbar_kwargs)
cbar.outline.set_color('none')
cbar.update_ticks()
cbar.ax.xaxis.set_tick_params(labelsize=10)

ax1.set_title(' MME trends of T$_{cir}$ in future scenario',loc='center',color='black',fontsize=13,y=1.03)
ax1.text(0.06,1.1, 'a', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax1.transAxes)
# ax1.set_title('JA',loc='right',color='black',fontsize=13,y=1.03)

ax1.set_xlabel(' ')
ax1.set_ylabel(' ')
ax1.set_aspect(1.5)


#---------------------- 2.  ------------------------

#设置色标的
u_cbar_kwargs = {
    'orientation': 'horizontal',  # 改为水平方向
    'label': 'm/s/decade',
    'ticks': np.arange(-0.3, 0.3 + 0.1, 0.1),
    'pad': 0.05,
    'shrink': 0.65,
    'extend': 'both',
    'extendfrac': 0.1  
}


# #---------------------- 2. 绘制趋势结果 ------------------------

Colors=('#2c0248',"#450e69",'#5f3a91','#8477af','#ada6ce','#cfcfe5','#ffffff','#ffffff','#fde1b5','#fab35b','#e2861a','#bd6109',"#985524",'#75370c')
levels = np.arange(-0.3,  0.3 + 0.05, 0.05)

fut_u_anom_grd = xr.DataArray(data= future_model_U, dims=['lat','lon' ],
            coords= {'lat':range(12,68+2,2),'lon':range(0,358+2,2)})
fut_u_anom_grd = mask_land(fut_u_anom_grd,'ocean','lon') 

global_t_composite, cycle_lon = add_cyclic_point(fut_u_anom_grd, coord=lon1) 
mappable = ax2.contourf(range(0,358+2+2,2),range(12,68+2,2), global_t_composite,transform=ccrs.PlateCarree(),colors = Colors,levels=levels,extend='both')



c1p = ax2.contourf( range(0,358+2,2),range(12,68+2,2), fut_Model_U_test, levels =[0,13.5,20.0],hatches=[ None,'XXX'],colors="none", transform=ccrs.PlateCarree())


for collection in c1p.collections:
    collection.set_edgecolor('white')
for collection in c1p.collections:
    collection.set_linewidth(0)

new_position = fig.add_axes([0.89, 0.85,0.35, 0.011 ]) 
cbar=fig.colorbar(mappable,norm=norm,  cax=new_position,**u_cbar_kwargs)
cbar.outline.set_color('none')
cbar.ax.xaxis.set_tick_params(labelsize=10)

ax2.set_title(' MME trends of TWAI in future scenario',loc='center',color='black',fontsize=13,y=1.03)
ax2.text(0.06,1.1, 'b', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax2.transAxes)
     
ax2.set_xlabel(' ')
ax2.set_ylabel(' ')
ax2.set_aspect(1.5)


ax2.axis('off') 
ax2.add_feature(cfeature.COASTLINE,  linewidth=0.5 , edgecolor='black')
ax2.add_feature(cfeature.BORDERS, linestyle = 'dotted',lw = 0.35, edgecolor='black')





# 绘制一个矩形框(NE US)


for k in range(1,3):

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
        (88, 30),  # 左下角坐标 (longitude, latitude)
        18, 15, # 宽度和高度 (longitude, latitude)
        linewidth=1.3,edgecolor='black',
        facecolor='none',linestyle='--',
        transform=ccrs.PlateCarree()
    )
    locals()['ax' + str(k)].add_patch(locals()['rectangle' + str(k)])  

    # #North Asia
    # locals()['rectangle' + str(k)] = Rectangle(
    #     (135, 58), 
    #     40, 11, # 宽度和高度 (longitude, latitude)
    #     linewidth=1.3,edgecolor='black',
    #     facecolor='none',linestyle='--',
    #     transform=ccrs.PlateCarree()
    # )
    # locals()['ax' + str(k)].add_patch(locals()['rectangle' + str(k)]) 



###----------------------------------------------------------------
### double box
###----------------------------------------------------------------


data_his = {"EE":his_EE_trend,
            "CC":his_CA_trend,
            # "ES":his_NA_trend,
            "WNA":his_WNA_trend} ; data_his = pd.DataFrame(data_his) 

data_ssp = {"EE":fut_EE_trend,
            "CC":fut_CA_trend,
            # "ES":fut_NA_trend,
            "WNA":fut_WNA_trend} ; data_ssp = pd.DataFrame(data_ssp) 

x1 = np.arange(1,9,3) 
x2 = x1+1

box1 = ax5.boxplot(data_his.dropna(), positions=x1, patch_artist=True,showmeans=True,
            boxprops={"facecolor": "#febe50",
                      "edgecolor": "none",
                      "linewidth": 0.5},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker':'+',
                       'markerfacecolor':'k',
                       'markeredgecolor':'k',
                       'markersize':6},
             capprops={'color': "#febe50", 'linewidth': 1},
             flierprops={'marker': 'o','markerfacecolor':'none','markeredgecolor':"#febe50"},
              whiskerprops={'color':"#febe50",'linestyle':'--'})

box2 = ax5.boxplot(data_ssp.dropna(),positions=x2,patch_artist=True,showmeans=True,
            boxprops={"facecolor": "#b14a42",
                      "edgecolor": "none",
                      "linewidth": 0.5},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker':'+',
                       'markerfacecolor':'k',
                       'markeredgecolor':'k',
                       'markersize':6},
            capprops={'color': "#b14a42", 'linewidth': 1},
            flierprops={'marker': 'o','markerfacecolor':'none','markeredgecolor':"#b14a42",},
            whiskerprops={'color': "#b14a42",'linestyle':'--'})

city = ["EE","NWC","WNA"]
ax5.set_xticks([1.5,4.5,7.5],city,fontsize=11)
ax5.set_xticklabels(city)  #,rotation=30)
ax5.set_title('Estimated T$_{cir}$ trends by TWAI',loc='center',color='black',fontsize=13,y=1.03)
ax5.text(-0.115,1.1, 'c', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax5.transAxes)
     
ax5.axhline(y=0 , color='gray', linestyle='-',linewidth=3, alpha=0.5,zorder=0)
# ax5.set_ylim(-0.1,0.3)
ax5.set_ylabel('Estimated T$_{cir}$ trend (°C/decade)',fontsize=11)
ax5.tick_params(axis='both', which='major', labelsize=11)

# 给箱体添加图例，每类箱线图中取第一个颜色块用于代表图例
# ax5 = plt.gca() #返回坐标轴
ax5.legend(handles=[box1['boxes'][0],box2['boxes'][0]],labels=['1979-2023','2055-2099'], frameon=False)


###----------------------------------------------------------------
### double box
###----------------------------------------------------------------

import seaborn as sns
from scipy.stats import linregress
def scatter_plot(ax,x,y,title,x_title,y_title ):
 
    # 计算回归直线
    slope1, intercept1, r_value1, p_value1, std_err1 = linregress(x, y)
    regression_line = slope1 * x + intercept1
    print(r_value1, p_value1, std_err1)

    ax.axhline(0.0,  0,1,color='darkgray', linestyle=':', linewidth=2,alpha=0.6,zorder=1)
    ax.axvline(0.0, color='darkgray', linestyle=':', linewidth=2,alpha=0.6,zorder=1)


    # sc = ax.scatter(x, y, edgecolors='k',  alpha=0.75) #
    # 
    # sns.regplot(x=x,y=y,line_kws={'color':'#f19e7d','alpha':0.5},ci=95,ax=ax,scatter=False)
    ax.plot(x, regression_line, color='#ba2832',linewidth=2, linestyle='-')#,label=f"R:{r_value1:.2f} (p={p_value1:.2g})"
    
  

    # 添加 
    ax.tick_params(axis='both',labelsize=9, length = 3,direction = 'out', width=0.49)
    ax.set_xlabel(x_title,fontsize=11)

    ax.set_ylabel(y_title,fontsize=11)
    ax.set_ylim( 0,0.55)
    ax.set_xlim(0.03,0.5)
    ax.set_title(title,loc='left',color='black',fontsize=13,y=1.03)
    ax.legend(fontsize=10,frameon=False)

 
model_name=("MME mean","ACCESS-CM2","CanESM5","EC-Earth3","EC-Earth3-Veg",
            "FGOALS-g3","GFDL-CM4","HadGEM3-GC31-LL",\
            "HadGEM3-GC31-MM","INM-CM4-8","INM-CM5-0","IPSL-CM6A-LR",\
            "KACE-1-0-G","MIROC6","MPI-ESM1-2-HR","MPI-ESM1-2-LR",\
            "MRI-ESM2-0","NorESM2-LM","NorESM2-MM","TaiESM1","UKESM1-0-LL") #24s


model_marker = ['s','^','o','s','D','*','x', 
        '^','x','s','*','D','o', \
        '^','o','s','D','*','x', \
        '^','o','s','D','*','x', \
        '^','o','s','x','*','D']
model_color = ['black','red','red','#ef4c96','#8d0404','pink','orange','orange',\
       'blue','blue','#151594','#a42af1','purple','#2091ff','#3eff3e','#3eff3e',\
       '#006400','#006400','#98fb98','#c2ff46','#7f7f7f']
model_size =  [9,6.5,6.5,6.3,5.5,7.5,6.5,7.0, 
        6.5,6.5,7.5,6.5,6.5,7.0,6.5, \
        6.5,6.5,7.5,6.5,6.8,6.5]

eddy_future_T_trend = future_T_trend - np.tile(np.mean(future_T_trend, axis =2),(180,1,1)).transpose(1,2,0)
TWI_future=  np.mean(future_U_trend.loc[:,42:42+12,235:235+15],axis=(1,2))
T_future= np.mean(future_T_trend.loc[:,42:42+12,235:235+15],axis=(1,2))

All_TWI_future = np.zeros((21))
All_T_future = np.zeros((21))

All_TWI_future[0] = np.mean(TWI_future)
All_TWI_future[1:21] = TWI_future
All_T_future[0] = np.mean(T_future)
All_T_future[1:21] = T_future

im1= scatter_plot(ax6,TWI_future,T_future,
             title="Trends of TWAI and T$_{cir}$ in western North America " ,x_title="TWAI trends",
             y_title="T$_{cir}$ trends")


slope1, intercept1, r_value1, p_value1, std_err1 = linregress(TWI_future,T_future)
print(r_value1, p_value1)

for imodel in range(21):
    ax6.plot(All_TWI_future[imodel], All_T_future[imodel], label=model_name[imodel], marker=model_marker[imodel],
        markerfacecolor='none', markeredgecolor=model_color[imodel], markeredgewidth=1.5, linewidth=0.0, markersize=model_size[imodel])
           

# ax6.legend(fontsize=10,frameon=False)
ax6.legend(loc='lower left',fontsize=10,frameon=False, bbox_to_anchor=(-1.6, -0.28),
              fancybox=True, shadow=True, ncol=3)

plt.tight_layout()
ax6.tick_params(axis='both', which='major', labelsize=11)
ax6.text(-0.115,1.05, 'd', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax6.transAxes)
ax6.text(0.08,0.85, f'r = {r_value1:.2f}', fontsize=13, fontfamily='sans-serif',color='#0073bd',transform=ax6.transAxes)
ax6.text(0.08,0.76, '(p<0.01)', fontsize=13, fontfamily='sans-serif',color='#0073bd',transform=ax6.transAxes)
 

plt.savefig("/public/home/fcai/abc/2Paper_blocking/Final_figures/Figure5.pdf", dpi=500, bbox_inches='tight')
plt.show()