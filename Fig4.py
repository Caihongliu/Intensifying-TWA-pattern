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
from pylab import *
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
module_path = '/public/master/data1/home/fcai/abc/1Paper_2023/data0'
sys.path.append(module_path)
import glob
import os

from scipy.stats import pearsonr


def runavg(x, width):
 #输入一原始时间序列和宽度，返回一个平滑以后的时间序列
    n = len(x)
    x = np.append(x, np.append(x, x))
    x_smooth = np.convolve(x, np.ones(width)/width, mode='valid')
    xs = x_smooth[n:2*n]
    return xs


# =========================
# 1. 基础函数
# =========================
def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def weighted_cov(x, y, w):
    mx = weighted_mean(x, w)
    my = weighted_mean(y, w)
    return np.sum(w * (x - mx) * (y - my)) / np.sum(w)

def weighted_corr(x, y, w):
    cov_xy = weighted_cov(x, y, w)
    var_x = weighted_cov(x, x, w)
    var_y = weighted_cov(y, y, w)
    if var_x <= 0 or var_y <= 0:
        return np.nan
    return cov_xy / np.sqrt(var_x * var_y)

def pearson_significance_from_neff(r, neff):
    """
    用有效自由度 neff 做 Pearson t-test
    """
    if np.isnan(r) or neff is None or neff <= 2:
        return np.nan, np.nan

    # 防止 r=±1 数值溢出
    r = np.clip(r, -0.999999, 0.999999)
    tval = r * np.sqrt((neff - 2) / (1 - r**2))
    pval = 2 * (1 - stats.t.cdf(np.abs(tval), df=neff - 2))
    return tval, pval
    
# =========================
# 2. 空间有效自由度估计
# =========================
def lag1_corr_1d(a, b):
 
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 3:
        return np.nan
    aa = a[mask]
    bb = b[mask]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return np.nan
    return np.corrcoef(aa, bb)[0, 1]


def spatial_lag1_autocorr(field2d):

    f = np.array(field2d, dtype=float)

    # 经向（上下相邻）
    r_merid = lag1_corr_1d(f[:-1, :].ravel(), f[1:, :].ravel())

    # 纬向（左右相邻）
    r_zonal = lag1_corr_1d(f[:, :-1].ravel(), f[:, 1:].ravel())

    vals = [r for r in [r_merid, r_zonal] if np.isfinite(r)]
    if len(vals) == 0:
        return np.nan
    return np.mean(vals)


def effective_sample_size_spatial(x2d, y2d, w2d=None):
    mask = np.isfinite(x2d) & np.isfinite(y2d)
    if np.sum(mask) < 3:
        return np.nan

    x = np.where(mask, x2d, np.nan)
    y = np.where(mask, y2d, np.nan)

    # --- 第一步：名义样本量 n0 ---
    if w2d is None:
        n0 = np.sum(mask)
    else:
        ww = np.where(mask, w2d, 0.0)
        sw = np.sum(ww)
        sw2 = np.sum(ww**2)
        if sw <= 0 or sw2 <= 0:
            return np.nan
        # Kish effective sample size for unequal weights
        n0 = (sw**2) / sw2

    # --- 第二步：空间自相关修正 ---
    rho_x = spatial_lag1_autocorr(x)
    rho_y = spatial_lag1_autocorr(y)

    # 如果估不出来，就退化成 n0
    if not np.isfinite(rho_x) or not np.isfinite(rho_y):
        return max(3, n0)

    rho_prod = rho_x * rho_y

    # 避免奇异情况
    rho_prod = np.clip(rho_prod, -0.99, 0.99)

    neff = n0 * (1 - rho_prod) / (1 + rho_prod)

    # 边界保护
    neff = max(3, min(neff, n0))
    return neff


# =========================
# 3. 加权空间相关
# =========================
def build_lat_weights(lat, shape2d):
    lat = np.asarray(lat)

    if lat.ndim == 1:
        # shape2d = (nlat, nlon)
        w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, shape2d[1]))
    elif lat.ndim == 2:
        w = np.cos(np.deg2rad(lat))
    else:
        raise ValueError("lat 必须是一维或二维数组")

    return w


def spatial_corr_weighted(field1, field2, lat=None):
    x2d = np.array(field1, dtype=float)
    y2d = np.array(field2, dtype=float)

    if x2d.shape != y2d.shape:
        raise ValueError("两个空间场的 shape 必须一致")

    mask = np.isfinite(x2d) & np.isfinite(y2d)
    n_raw = np.sum(mask)

    if n_raw < 3:
        return np.nan, n_raw, np.nan, np.nan

    # 权重
    if lat is None:
        w2d = np.ones_like(x2d, dtype=float)
    else:
        w2d = build_lat_weights(lat, x2d.shape)

    w = w2d[mask]
    x = x2d[mask]
    y = y2d[mask]

    r = weighted_corr(x, y, w)
    n_eff = effective_sample_size_spatial(x2d, y2d, w2d=w2d)
    _, p = pearson_significance_from_neff(r, n_eff)

    return r, n_raw, n_eff, p

import numpy as np

def spatial_corr(A, B):
    a = np.asarray(A).ravel()
    b = np.asarray(B).ravel()

    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]

    if len(a) < 2:
        return np.nan

    return np.corrcoef(a, b)[0, 1]



ds0=xr.open_dataset("/public/master/data1/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/Function/U300Positive_significant_995.nc")
lat=ds0.lat1; lon=ds0.lon1
network=ds0.significant_99
print(network)

##--##--## 计算不同地区 --##--##--##


concur_WNA = np.sum(network.loc[40:40+12,-125:-110,:,:], axis=(0,1))     
concur_EE  = np.sum(network.loc[45:45+20,18:18+38,:,:], axis=(0,1)) 
concur_CA  = np.sum(network.loc[33:33+15,88:88+18,:,:], axis=(0,1)) 
concur_NA  = np.sum(network.loc[58:58+11,135:135+40,:,:], axis=(0,1)) 


network_WNA = (concur_WNA)
network_EE = (concur_EE)
network_CA = (concur_CA)
network_NA = (concur_NA)

# network_all = network_WNA+network_EE+network_CA+network_NA
# 变成矢量

network_EE = xr.DataArray(data= network_EE, dims=['lat','lon'],  
            coords= {'lat':lat.data , 'lon':lon.data})    ; print(network_EE.shape)
network_CA = xr.DataArray(data= network_CA, dims=['lat','lon'],
            coords= {'lat':lat.data , 'lon':lon.data})
network_NA = xr.DataArray(data= network_NA, dims=['lat','lon'],
                coords= {'lat':lat.data , 'lon':lon.data})
network_WNA = xr.DataArray(data= network_WNA, dims=['lat','lon'],
            coords= {'lat':lat.data , 'lon':lon.data})
# network_All = xr.DataArray(data= network_all, dims=['lat','lon'],
#             coords= {'lat':lat.data , 'lon':lon.data})

 

print("=====================================")
corr_EE_NWC, n_raw, n_eff, p = spatial_corr_weighted(network_EE.data, network_CA.data, lat=lat)
print(f"corr_EE_NWC Weighted spatial correlation: {corr_EE_NWC:.2f}")
print(f"Raw n = {n_raw}")
print(f"Effective n = {n_eff:.2f}")
print(f"Adjusted p-value: {p:.8f}")

print("=====================================")
corr_NWC_ES, n_raw, n_eff, p = spatial_corr_weighted(network_CA.data, network_NA.data, lat=lat)
print(f"corr_NWC_ES Weighted spatial correlation: {corr_NWC_ES:.2f}")
print(f"Raw n = {n_raw}")
print(f"Effective n = {n_eff:.2f}")
print(f"Adjusted p-value: {p:.8f}")

print("=====================================")
corr_EE_ES, n_raw, n_eff, p = spatial_corr_weighted(network_NA.data, network_EE.data, lat=lat)
print(f"corr_EE_ES Weighted spatial correlation: {corr_EE_ES:.2f}")
print(f"Raw n = {n_raw}")
print(f"Effective n = {n_eff:.2f}")
print(f"Adjusted p-value: {p:.8f}")

print("=====================================")
corr_EE_WNA, n_raw, n_eff, p = spatial_corr_weighted(network_WNA.data, network_EE.data, lat=lat)
print(f"corr_EE_WNA Weighted spatial correlation: {corr_EE_WNA:.2f}")
print(f"Raw n = {n_raw}")
print(f"Effective n = {n_eff:.2f}")
print(f"Adjusted p-value: {p:.8f}")

print("=====================================")
corr_NWC_WNA, n_raw, n_eff, p = spatial_corr_weighted(network_WNA.data, network_CA.data, lat=lat)
print(f"corr_NWC_WNA Weighted spatial correlation: {corr_NWC_WNA:.2f}")
print(f"Raw n = {n_raw}")
print(f"Effective n = {n_eff:.2f}")
print(f"Adjusted p-value: {p:.8f}")

print("=====================================")
corr_ES_WNA, n_raw, n_eff, p = spatial_corr_weighted(network_WNA.data, network_NA.data, lat=lat)
print(f"corr_ES_WNA Weighted spatial correlation: {corr_ES_WNA:.2f}")
print(f"Raw n = {n_raw}")
print(f"Effective n = {n_eff:.2f}")
print(f"Adjusted p-value: {p:.8f}")





# Easterly (right column)
ds0_E=xr.open_dataset("/public/master/data1/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/Function/U300Negative_significant_995.nc")

lat_E=ds0.lat1; lon_E=ds0.lon1
network_E=ds0_E.significant_99
print(network_E)

##--##--## 计算不同地区 --##--##--##

concur_WNA_E = np.sum(network_E.loc[40:40+12,-125:-110,:,:], axis=(0,1))     
concur_EE_E  = np.sum(network_E.loc[45:45+20,18:18+38,:,:], axis=(0,1)) 
concur_CA_E  = np.sum(network_E.loc[33:33+15,88:88+18,:,:], axis=(0,1)) 
concur_NA_E  = np.sum(network_E.loc[58:58+11,135:135+40,:,:], axis=(0,1)) 



network_WNA_E = (concur_WNA_E)
network_EE_E = (concur_EE_E)
network_CA_E = (concur_CA_E)
network_NA_E = (concur_NA_E)

network_all_E = network_WNA_E+network_EE_E+network_CA_E+network_NA_E
# 变成矢量

network_EE_E = xr.DataArray(data= network_EE_E, dims=['lat_E','lon_E'],
            coords= {'lat_E':lat_E.data , 'lon_E':lon_E.data})
network_CA_E = xr.DataArray(data= network_CA_E, dims=['lat_E','lon_E'],
            coords= {'lat_E':lat_E.data , 'lon_E':lon_E.data})
network_NA_E = xr.DataArray(data= network_NA_E, dims=['lat_E','lon_E'],
                coords= {'lat_E':lat_E.data , 'lon_E':lon_E.data})
network_WNA_E = xr.DataArray(data= network_WNA_E, dims=['lat_E','lon_E'],
            coords= {'lat_E':lat_E.data , 'lon_E':lon_E.data})
network_All_E = xr.DataArray(data= network_all_E, dims=['lat_E','lon_E'],
            coords= {'lat_E':lat_E.data , 'lon_E':lon_E.data})


print("=====================================")
corr_EE_NWC_E, n_raw, n_eff, p = spatial_corr_weighted(network_EE_E.data, network_CA_E.data, lat=lat_E)
print(f"corr_EE_NWC_E Weighted spatial correlation: {corr_EE_NWC_E:.2f}")
print(f"Raw n = {n_raw}")
print(f"Effective n = {n_eff:.2f}")
print(f"Adjusted p-value: {p:.8f}")

print("=====================================")
corr_NWC_ES_E, n_raw, n_eff, p = spatial_corr_weighted(network_CA_E.data, network_NA_E.data, lat=lat_E)
print(f"corr_NWC_ES_E Weighted spatial correlation: {corr_NWC_ES_E:.2f}")
print(f"Raw n = {n_raw}")
print(f"Effective n = {n_eff:.2f}")
print(f"Adjusted p-value: {p:.8f}")

print("=====================================")
corr_EE_ES_E, n_raw, n_eff, p = spatial_corr_weighted(network_NA_E.data, network_EE_E.data, lat=lat_E)
print(f"corr_EE_ES_E Weighted spatial correlation: {corr_EE_ES_E:.2f}")
print(f"Raw n = {n_raw}")
print(f"Effective n = {n_eff:.2f}")
print(f"Adjusted p-value: {p:.8f}")

print("=====================================")
corr_EE_WNA_E, n_raw, n_eff, p = spatial_corr_weighted(network_WNA_E.data, network_EE_E.data, lat=lat_E)
print(f"corr_EE_WNA_E Weighted spatial correlation: {corr_EE_WNA_E:.2f}")
print(f"Raw n = {n_raw}")
print(f"Effective n = {n_eff:.2f}")
print(f"Adjusted p-value: {p:.8f}")

print("=====================================")
corr_NWC_WNA_E, n_raw, n_eff, p = spatial_corr_weighted(network_WNA_E.data, network_CA_E.data, lat=lat_E)
print(f"corr_NWC_WNA_E Weighted spatial correlation: {corr_NWC_WNA_E:.2f}")
print(f"Raw n = {n_raw}")
print(f"Effective n = {n_eff:.2f}")
print(f"Adjusted p-value: {p:.8f}")

print("=====================================")
corr_ES_WNA_E, n_raw, n_eff, p = spatial_corr_weighted(network_WNA_E.data, network_NA_E.data, lat=lat_E)
print(f"corr_ES_WNA_E Weighted spatial correlation: {corr_ES_WNA_E:.2f}")
print(f"Raw n = {n_raw}")
print(f"Effective n = {n_eff:.2f}")
print(f"Adjusted p-value: {p:.8f}")




#----------------------分割线（画图函数）------------------------
with open('/public/master/data1/home/fcai/abc/data/CN-border-La.dat') as src:
    context = src.read()
    blocks = [cnt for cnt in context.split('>') if len(cnt) > 0]
    borders = [np.fromstring(block, dtype=float, sep=' ') for block in blocks]


fig = plt.figure(figsize=[16, 12],frameon=True)
ax1 = fig.add_axes([0.02, 0.82, 0.3, 0.1], projection=ccrs.PlateCarree(central_longitude=163) )
ax2 = fig.add_axes([0.02, 0.6, 0.3, 0.1], projection=ccrs.PlateCarree(central_longitude=163) )
ax3 = fig.add_axes([0.02, 0.38, 0.3, 0.1], projection=ccrs.PlateCarree(central_longitude=163) )
ax4 = fig.add_axes([0.02, 0.16, 0.3, 0.1], projection=ccrs.PlateCarree(central_longitude=163) )
ax5 = fig.add_axes([0.36, 0.82, 0.3, 0.1], projection=ccrs.PlateCarree(central_longitude=163) )
ax6 = fig.add_axes([0.36, 0.6, 0.3, 0.1], projection=ccrs.PlateCarree(central_longitude=163) )
ax7 = fig.add_axes([0.36, 0.38, 0.3, 0.1], projection=ccrs.PlateCarree(central_longitude=163) )
ax8 = fig.add_axes([0.36, 0.16, 0.3, 0.1], projection=ccrs.PlateCarree(central_longitude=163) )



box = [-180, 180, 10, 80]
scale = '50m'            
xstep, ystep = 90, 35

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
                    labelsize=12,
                    bottom=True, left=True, right=False, top=False)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
    return ax


make_map_Eurasia(ax1)
make_map_Eurasia(ax2)
make_map_Eurasia(ax3)
make_map_Eurasia(ax4)
make_map_Eurasia(ax5)
make_map_Eurasia(ax6)
make_map_Eurasia(ax7)
make_map_Eurasia(ax8)

def make_composite(composite,lat,lon,ax,title,levels,Colors=None):

    cbar_kwargs = {
        'orientation': 'horizontal',  # 改为水平方向
        'label': '',
        'ticks':levels,
        'pad': 0.05,
        'shrink': 0.65,
        'extend': 'both',
        'extendfrac': 0.1  
    }

    levels = levels
    if Colors is None:
        Colors=('#ffffff','#ffe101','#feaa00',\
                '#ff8000','#ff3234','#da0003') 


            
    global_t_composite2, cycle_lon = add_cyclic_point(composite, coord=lon)   ## 给数据加上 360度经度 位置的数据
    c=ax.contourf(cycle_lon,lat ,global_t_composite2, cbar_kwargs=cbar_kwargs,transform=ccrs.PlateCarree(),colors=Colors,levels=levels,extend='both')
    ax.set_title(title,loc='center',color='black',fontsize=13,y=1.03)
    ax.set_xlabel(' ')
    ax.set_ylabel(' ')
    ax.set_aspect(1.3) 
    ax.add_feature(cfeature.BORDERS, linestyle = 'dotted',lw = 0.35, edgecolor='black')

    return c


c1=make_composite(network_EE,lat,lon,ax1,'Westerly event related to EE heatwaves',levels=[500,1000,2000,3000,5000 ]) 
c2=make_composite(network_CA,lat,lon,ax2,'Westerly event related to NWC heatwaves',levels=[200, 500, 1000,1500, 3000 ])
c3=make_composite(network_NA,lat,lon,ax3,'Westerly event related to ES heatwaves',levels=[500, 1000,2000, 3000, 5000])
c4=make_composite(network_WNA,lat,lon,ax4,'Westerly event related to WNA heatwaves',levels=[200, 500, 1000,1500, 2000])

Colors_E=('#ffffff','#deebf7','#bce5f8','#8acbef','#4a91cc','#1f4182')
c5=make_composite(network_EE_E,lat_E,lon_E,ax5,'Easterly event related to EE heatwaves',levels=[500,1000,2000,3000,5000],Colors=Colors_E)
c6=make_composite(network_CA_E,lat_E,lon_E,ax6,'Easterly event related to NWC heatwaves',levels=[200, 500,1000,1500,3000 ],Colors=Colors_E)
c7=make_composite(network_NA_E,lat_E,lon_E,ax7,'Easterly event related to ES heatwaves',levels=[500, 1000,2000, 3000, 5000],Colors=Colors_E)
c8=make_composite(network_WNA_E,lat_E,lon_E,ax8,'Easterly event related to WNA heatwaves',levels=[200, 500, 1000,1500, 2000],Colors=Colors_E)


cax = fig.add_axes([0.04,0.77,0.23,0.011])
cbar=fig.colorbar(c1,cax = cax,ax=[ax1],orientation= 'horizontal', shrink=0.6 )
cbar.outline.set_color('none')
cbar.update_ticks()
cbar.set_ticks([500, 1000,2000,3000,5000])
cbar.set_ticklabels([5,10, 20,30,50])
cbar.ax.xaxis.set_tick_params(labelsize=10)

cax1 = fig.add_axes([0.04,0.55,0.23,0.011])
cbar1=fig.colorbar(c2,cax = cax1,ax=[ax2],orientation= 'horizontal' )
cbar1.outline.set_color('none')
cbar1.update_ticks()
cbar1.set_ticks([200, 500, 1000,1500, 3000])
cbar1.set_ticklabels([2, 5, 10,15, 30])
cbar1.ax.xaxis.set_tick_params(labelsize=10)

cax2 = fig.add_axes([0.04,0.33,0.23,0.011])
cbar2=fig.colorbar(c3,cax = cax2,ax=[ax3],orientation= 'horizontal', shrink=0.6 )
cbar2.outline.set_color('none')
cbar2.update_ticks()
cbar2.set_ticks([500, 1000,2000,3000,5000])
cbar2.set_ticklabels([5,10, 20,30,50])
cbar2.ax.xaxis.set_tick_params(labelsize=10)

cax4 = fig.add_axes([0.04,0.11,0.23,0.011])
cbar4=fig.colorbar(c4,cax = cax4,ax=[ax4],orientation= 'horizontal', shrink=0.6 )
cbar4.outline.set_color('none')
cbar4.update_ticks()
cbar4.set_ticks([200, 500, 1000,1500, 2000])
cbar4.set_ticklabels([2, 5, 10,15, 20])
cbar4.ax.xaxis.set_tick_params(labelsize=10)



cax5 = fig.add_axes([0.38,0.77,0.23,0.011])
cbar5=fig.colorbar(c5,cax = cax5,ax=[ax5],orientation= 'horizontal', shrink=0.6 )
cbar5.outline.set_color('none')
cbar5.update_ticks()
cbar5.set_ticks([500, 1000,2000,3000,5000])
cbar5.set_ticklabels([5,10, 20,30,50])
cbar5.ax.xaxis.set_tick_params(labelsize=10)

cax6 = fig.add_axes([0.38,0.55,0.23,0.011])
cbar6=fig.colorbar(c6,cax = cax6,ax=[ax6],orientation= 'horizontal' )
cbar6.outline.set_color('none')
cbar6.update_ticks()
cbar6.set_ticks([200, 500, 1000,1500, 3000])
cbar6.set_ticklabels([2, 5, 10,15, 30])
cbar6.ax.xaxis.set_tick_params(labelsize=10)

cax7 = fig.add_axes([0.38,0.33,0.23,0.011])
cbar7=fig.colorbar(c7,cax = cax7,ax=[ax7],orientation= 'horizontal', shrink=0.6 )
cbar7.outline.set_color('none')
cbar7.update_ticks()
cbar7.set_ticks([500, 1000,2000,3000,5000])
cbar7.set_ticklabels([5,10, 20,30,50])
cbar7.ax.xaxis.set_tick_params(labelsize=10)

cax8 = fig.add_axes([0.38,0.11,0.23,0.011])
cbar8=fig.colorbar(c8,cax = cax8,ax=[ax8],orientation= 'horizontal', shrink=0.6 )
cbar8.outline.set_color('none')
cbar8.update_ticks()
cbar8.set_ticks([200, 500, 1000,1500, 2000])
cbar8.set_ticklabels([2, 5, 10,15, 20])
cbar8.ax.xaxis.set_tick_params(labelsize=10)








ax1.text(0,1.15, 'a', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax1.transAxes)
ax2.text(0,1.15, 'c', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax2.transAxes)
ax3.text(0,1.15, 'e', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax3.transAxes)
ax4.text(0,1.15, 'g', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax4.transAxes)
ax5.text(0,1.15, 'b', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax5.transAxes)
ax6.text(0,1.15, 'd', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax6.transAxes)
ax7.text(0,1.15, 'f', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax7.transAxes)
ax8.text(0,1.15, 'h', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax8.transAxes)


# 绘制一个矩形框(NE Asia)


#East Europe
def create_east_europe_rectangle(color='blue'):
    return Rectangle(
        (18, 45),
        38, 20, 
        linewidth=1.5, edgecolor=color,
        facecolor='none',
        transform=ccrs.PlateCarree()
    )

ax1.add_patch(create_east_europe_rectangle())
ax5.add_patch(create_east_europe_rectangle('#fe0000'))
 
#Central Asia
def create_northwest_china_rectangle(color='blue'):
    return Rectangle(
        (88, 33),  # 左下角坐标 (longitude, latitude)
        18, 15, # 宽度和高度 (longitude, latitude)
        linewidth=1.5,edgecolor=color,
        facecolor='none',
        transform=ccrs.PlateCarree()
    )

ax2.add_patch(create_northwest_china_rectangle())
ax6.add_patch(create_northwest_china_rectangle('#fe0000'))
 
#North Asia
def create_eastern_siberia_rectangle(color='blue'):
    return Rectangle(
        (135, 58), 
        40, 11, 
        linewidth=1.5,edgecolor=color,
        facecolor='none',
        transform=ccrs.PlateCarree()
    )
ax3.add_patch(create_eastern_siberia_rectangle())
ax7.add_patch(create_eastern_siberia_rectangle('#fe0000')) 

#West NA
def create_western_america_rectangle(color='blue'):
    return Rectangle(
        (235, 40), 
        15, 12, 
        linewidth=1.5,edgecolor=color,
        facecolor='none',
        transform=ccrs.PlateCarree()
    )
ax4.add_patch(create_western_america_rectangle())
ax8.add_patch(create_western_america_rectangle('#fe0000')) 


for i in range(1, 9):
    ax = locals()['ax' + str(i)]
    
    ax.plot(
        [242, 242],      # 经度不变 → 竖线
        [10, 80],        # 纬度范围
        linestyle='--',
        color='#cdd0cd',
        linewidth=2,
        transform=ccrs.PlateCarree()
    )

    ax.plot(
        [38, 38],      # 经度不变 → 竖线
        [10, 80],        # 纬度范围
        linestyle='--',
        color='#cdd0cd',
        linewidth=2,
        transform=ccrs.PlateCarree()
    )
  
    
    ax.plot(
        [97, 97],      # 经度不变 → 竖线
        [10, 80],        # 纬度范围
        linestyle='--',
        color='#cdd0cd',
        linewidth=2,
        transform=ccrs.PlateCarree()
    )

    ax.plot(
        [155, 155],      # 经度不变 → 竖线
        [10, 80],        # 纬度范围
        linestyle='--',
        color='#cdd0cd',
        linewidth=2,
        transform=ccrs.PlateCarree()
    )

     



plt.savefig("/public/master/data1/home/fcai/abc/2Paper_blocking/Final_figures/Last_sig99_network.pdf", dpi=300, bbox_inches='tight')
plt.show()
