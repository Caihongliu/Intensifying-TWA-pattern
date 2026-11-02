from cartopy.util import add_cyclic_point
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from copy import copy
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import shapely.geometry as sgeom
import matplotlib.colorbar as colorbar
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from matplotlib.patches import Rectangle,Polygon
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
from global_land_mask import globe



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
 
# make_map_Eurasia(ax1) 

def spatial_corr(field1, field2):
    """计算两个二维场的空间相关系数"""
    # 展平为1维数组
    f1 = field1.flatten()
    f2 = field2.flatten()
    # 去除NaN值（如果有）
    mask = ~np.isnan(f1) & ~np.isnan(f2)
    f1 = f1[mask]
    f2 = f2[mask]
    # 计算相关系数
    return np.corrcoef(f1, f2)[0, 1]

def pearson_significance(r, n):
    """传统t检验(忽略空间自相关）"""
    from scipy.stats import t
    t_stat = r * np.sqrt((n-2) / (1-r**2))
    p_value = 2 * t.sf(np.abs(t_stat), df=n-2)  # 双尾检验
    return p_value

def trend_result(data_3D,nt,nlat,nlon):
    ##--##--## 计算趋势 --##--##--##
    ngrd=nlat*nlon
    ano_grd = data_3D.reshape((nt, ngrd), order='F')
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

##--##--##--##--  创建温度数据  --##--##--##--##

ds1 = xr.open_dataset("/public/master/data1/home/fcai/abc/2Paper_blocking/historical_Dynamic/NEW_T_dyn_therm_anom_19502023_JJA_NH.nc")

ds1.coords['lon'] = np.mod(ds1['lon'], 360)
ds1 = ds1.reindex({ 'lon' : np.sort(ds1['lon'])})
 
T_dyn_anom = ds1.T_dyn_anom 



lat = ds1.lat.loc[0:70];   lon = ds1.lon 
year=T_dyn_anom.time.dt.year
month=T_dyn_anom.time.dt.month
 
## ------------- 1. 读取 动态气温数据 (1979-2023, 7-8月)
year_ind=(year>=1979)&(year<=2023)

T_dyn_Jul=T_dyn_anom.where(year_ind&(month==7),drop=True)
T_dyn_Aug=T_dyn_anom.where(year_ind&(month==8),drop=True)

T_dyn_4D=np.zeros((45,62,71,360))

for iyear in range(1979,2024):
    i=iyear-1979
    T_dyn_4D[i,0:31,:,:]=T_dyn_Jul[i*31+0:i*31+31,:,:]
    T_dyn_4D[i,31:62,:,:]=T_dyn_Aug[i*31+0:i*31+31,:,:]
   

##--##--##--##--  计算每个夏天的 75 分位阈值  --##--##--##--##
t_th75 = np.nanmean(T_dyn_4D, axis=1)    ## 计算mean

# np.percentile(T_dyn_4D, 75, axis=1)    ## 计算75%分位数值
# print(np.shape(t_th75))    ## 输出变量维度




##--##--## 计算趋势 --##--##--##
nt = 45;nlat=71;nlon=360
T_trend=trend_result(t_th75,nt,nlat,nlon)
p_values_grd = T_trend[1]
anom_grd = T_trend[0]
anom_grd = np.where(anom_grd==0,np.nan,anom_grd )

anom_grd = xr.DataArray(
    data=anom_grd,
    dims=[ "lat", "lon"],
    coords={ "lat": lat.values , "lon": lon.values}
)
p_values_grd = xr.DataArray(
    data=p_values_grd,
    dims=[ "lat", "lon"],
    coords={ "lat": lat.values , "lon": lon.values}
)
t_th75 = xr.DataArray(data= t_th75, dims=['year','lat','lon'],
              coords= {'year':np.linspace(1979,2023,45) ,'lat':lat.data , 'lon':lon.data})



## ------------- 2. multi-model mean of trend in Dyn T75

ds1 = xr.open_dataset("/public/master/data1/home/fcai/abc/2Paper_blocking/CMIP6_data/complex_network/6_pattern/data/Dynamical_Tmean_model_day_JA_19792023.nc") #"New_Dynamical_T75_model_day_JA_19502023.nc"
ds1.coords['lon'] = np.mod(ds1['lon'], 360)
ds1 = ds1.reindex({ 'lon' : np.sort(ds1['lon'])})
 
air_mean = ds1.air_th75;print(air_mean);print(air_mean)


Model_T_trend=np.zeros((20,71,360))

##--##--## 计算趋势 --##--##--##
nt = 45;nlat=71;nlon=360

for i in range(20):
    trend=trend_result(air_mean[i,:,:,:].to_numpy(),nt,nlat,nlon)
    Model_T_trend[i,:,:] = trend[0]

# ds6 = xr.open_dataset("/public/master/data1/home/fcai/abc/2Paper_blocking/Fig1/MMM_T_dyn_therm_anom_19792023_JA_NH.nc")
# Model_T_trend = ds6.Model_T_trend


Model_T_trend = xr.DataArray(data=Model_T_trend,
    dims=["Model", "lat", "lon"],
    coords={"Model":range(20), "lat": air_mean.lat.values , "lon": air_mean.lon.values}
)


MMM_T_anom = np.mean(Model_T_trend,0)
MMM_T_anom = xr.DataArray(
    data=MMM_T_anom,
    dims=["lat", "lon"],
    coords={"lat": air_mean.lat.values , "lon": air_mean.lon.values}
)

# # ##--##--##--##--  写进nc文件内  --##--##--##--##
# dss = xr.Dataset(data_vars=dict(Model_T_trend=Model_T_trend))
# output_dir = "/public/master/data1/home/fcai/abc/2Paper_blocking/historical_Dynamic/"
# dss.to_netcdf("/public/master/data1/home/fcai/abc/2Paper_blocking/Fig1/new_MMM_T_dyn_therm_anom_19792023_JA_NH.nc")

# # -------------- 验证 --------------
# T_test=np.zeros((20,71,360))
# T_test=np.where(Model_T_trend>0, 1, 0)
# T_test_Neg=np.where(Model_T_trend<0, 1, 0)


# ####20*90%=18
# p_t_test_pos=np.sum(T_test,0)
# p_t_test_neg=np.sum(T_test_Neg,0)


# -------------- 验证 --------------
# # 审稿人的 一致
# # -----------------------------------
Model_sign = np.sign(Model_T_trend)
Obser_sign = np.sign(anom_grd)

# 观测每个模式是否和观测一样
Agreement = np.where(Model_sign== np.expand_dims(Obser_sign,axis=0), 1,0)
Agree_num = np.sum(Agreement, axis=0)
# Agree_ratio=Agree_num/20.0


# # from scipy.stats import binom
# # from statsmodels.stats.multitest import fdrcorrection

# # nmodel=20

# # # 每个格点 得到 >=当前一致数量的概念; 每次成功概率0.5,看看他们的可能是不是随机的flunk
# # p_value = binom.sf(Agree_num -1, nmodel,0.5)


# # # FDR correction
# # reject, p_fdr = fdrcorrection(p_value.flatten(), alpha=0.1)
# # FDR_mask = reject.reshape(p_value.shape)

# # # 最终 inter-model agreement & 模式和观测的一致性
# # agreement_mask = (FDR_mask)



 

## ------------- 4. 四地 原温度和动态温度趋势的对比

w=np.cos(np.deg2rad(t_th75.lat))
print(np.shape(t_th75))

dyn_WNA = t_th75.loc[:,40:52,235:250].weighted(w).mean(dim=("lat","lon"))
dyn_EE  = t_th75.loc[:,45:65,18:56].weighted(w).mean(dim=("lat","lon"))
dyn_CA  = t_th75.loc[:,33:48,88:106].weighted(w).mean(dim=("lat","lon"))
dyn_NA  = t_th75.loc[:,58:69,135:175].weighted(w).mean(dim=("lat","lon"))

nt=45
dyn_WNA_trend = trend_1D_result(dyn_WNA.values ,nt)
dyn_EE_trend = trend_1D_result(dyn_EE.values ,nt)
dyn_CA_trend = trend_1D_result(dyn_CA.values ,nt)
dyn_NA_trend = trend_1D_result(dyn_NA.values ,nt)



#raw
 
ds2 = xr.open_dataset("/public/mgt/data2/model/fcai/data-observation/ERA5_daily/t2m/t2m.1979-01.daily.nc")
print(ds2)
t = ds2.t2m.loc[:,70:0,:][:,::-1,:]
lat = t.latitude; lon = t.longitude;print(lat) 

t2m = np.zeros((45,62,np.shape(t)[1],np.shape(t)[2]))
 
for iyear in range(45):
  print('iyear = ',iyear+1979)

  tJuly = xr.open_dataset("/public/mgt/data2/model/fcai/data-observation/ERA5_daily/t2m/t2m."+str(iyear+1979)+"-07.daily.nc")
  t2m[iyear,0:31,:,:] = tJuly.t2m.loc[:,70:0,:][:,::-1,:]
  tAugust = xr.open_dataset("/public/mgt/data2/model/fcai/data-observation/ERA5_daily/t2m/t2m."+str(iyear+1979)+"-08.daily.nc")
  t2m[iyear,31:62,:,:] = tAugust.t2m.loc[:,70:0,:][:,::-1,:]
  tJuly.close(); tAugust.close()

 

##--##--##--##--  计算每个夏天的 75 分位阈值  --##--##--##--##
t2m_th75 = np.mean(t2m, axis=1) #np.percentile(t2m, 75, axis=1)    ## 计算75%分位数值
print(np.shape(t2m_th75))    ## 输出变量维度

t2m_th75 = xr.DataArray(data= t2m_th75, dims=['year','lat','lon'],
              coords= {'year':np.linspace(1979,2023,45) ,'lat':lat.data , 'lon':lon.data})

w=np.cos(np.deg2rad(t2m_th75.lat))
# Raw
T_WNA = t2m_th75.loc[:,40:52,235:250].weighted(w).mean(dim=("lat","lon"))
T_EE  = t2m_th75.loc[:,45:65,18:56].weighted(w).mean(dim=("lat","lon"))
T_CA  = t2m_th75.loc[:,33:48,88:106].weighted(w).mean(dim=("lat","lon"))
T_NA  = t2m_th75.loc[:,58:69,135:175].weighted(w).mean(dim=("lat","lon"))

nt=45
T_WNA_trend = trend_1D_result(T_WNA ,nt)
T_EE_trend = trend_1D_result(T_EE ,nt)
T_CA_trend = trend_1D_result(T_CA ,nt)
T_NA_trend = trend_1D_result(T_NA ,nt)

print("dyn_WNA",dyn_WNA_trend,"      T_WNA",T_WNA_trend)# ,"      rate",(dyn_WNA_trend/T_WNA_trend ))
print("dyn_EE",dyn_EE_trend,"      T_EE",T_EE_trend)#,"      rate",str(dyn_EE_trend/T_EE_trend ))
print("dyn_CA",dyn_CA_trend,"      T_CA",T_CA_trend)#,"      rate",str(dyn_CA_trend/T_CA_trend ))
print("dyn_NA",dyn_NA_trend,"      T_NA",T_NA_trend)#,"      rate",str(dyn_NA_trend/T_NA_trend ))

print("      rate",(dyn_WNA_trend/T_WNA_trend ))
print("      rate",str(dyn_EE_trend/T_EE_trend ))
print("      rate",str(dyn_CA_trend/T_CA_trend ))
print("      rate",str(dyn_NA_trend/T_NA_trend ))



# Model


w = np.cos(np.deg2rad(Model_T_trend.lat))

Model_dyn_WNA = Model_T_trend.loc[:,40:52,235:250].weighted(w).mean(dim=("lat","lon"))
Model_dyn_EE  = Model_T_trend.loc[:,45:65,18:56].weighted(w).mean(dim=("lat","lon"))
Model_dyn_CA  = Model_T_trend.loc[:,33:48,88:106].weighted(w).mean(dim=("lat","lon"))
Model_dyn_NA  = Model_T_trend.loc[:,58:69,135:175].weighted(w).mean(dim=("lat","lon"))


#raw
 
# ds5 = xr.open_dataset("/public/master/data1/home/fcai/abc/2Paper_blocking/historical_Dynamic/Model_raw_T75_19502014_JA.nc")
ds5 = xr.open_dataset("/public/master/data1/home/fcai/abc/2Paper_blocking/historical_Dynamic/Model_raw_T75_19792023_JA.nc")
Model_raw_th75 = ds5.t2m_th75 
lat = ds5.lat; lon = ds5.lon


# Raw
w = np.cos(np.deg2rad(Model_raw_th75.lat))

Model_T_WNA = Model_raw_th75.loc[:,:,40:52,-125:-110].weighted(w).mean(dim=("lat","lon"))
Model_T_EE  = Model_raw_th75.loc[:,:,45:65,18:56].weighted(w).mean(dim=("lat","lon"))
Model_T_CA  = Model_raw_th75.loc[:,:,33:48,88:106].weighted(w).mean(dim=("lat","lon"))
Model_T_NA  = Model_raw_th75.loc[:,:,58:69,135:175].weighted(w).mean(dim=("lat","lon"))


nt=45
Model_raw_WNA_trend =np.zeros((20))
Model_raw_EE_trend =np.zeros((20))
Model_raw_CA_trend =np.zeros((20))
Model_raw_NA_trend =np.zeros((20))

 

for i in range(20):
    Model_raw_WNA_trend[i] = trend_1D_result(Model_T_WNA[i,:] ,nt)
    Model_raw_EE_trend[i] = trend_1D_result(Model_T_EE[i,:] ,nt)
    Model_raw_CA_trend[i] = trend_1D_result(Model_T_CA[i,:] ,nt)
    Model_raw_NA_trend[i] = trend_1D_result(Model_T_NA[i,:] ,nt)


#----------------------分割线（画图函数）------------------------
with open('/public/master/data1/home/fcai/abc/data/CN-border-La.dat') as src:
    context = src.read()
    blocks = [cnt for cnt in context.split('>') if len(cnt) > 0]
    borders = [np.fromstring(block, dtype=float, sep=' ') for block in blocks]


# Create a figure and axis

projection = ccrs.Robinson(central_longitude=150)

fig = plt.figure(figsize=[8, 12],frameon=True,dpi=300)
ax1 = fig.add_axes([0,0.7,1,0.7],projection=projection)
ax2 = fig.add_axes([0,0.4,1,0.7],projection=projection)
 

ax22 = fig.add_axes([1.11, 0.76,0.34, 0.4])


# # prepare 
box = [0, 360, 10, 70]  
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
                    length=8,
                    width=0.99, 
                    pad=3, 
                    labelsize=12,
                    bottom=True, left=True, right=False, top=False)
    ax.add_feature(cfeature.COASTLINE,  linewidth=0.4 , edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linestyle = 'dotted',lw = 0.25, edgecolor='black')
    return ax



def make_box(x,y,ax,colors,title,lat1,lat2):
    # add some docoration
    ax.plot(x, y,color = colors)
    # 坐标
    ax.tick_params(axis='both',labelsize=8, length = 3,direction = 'in', width=0.49)
    # # 画横线
    ax.axhspan(lat1, lat2, facecolor='#e8e8e8', alpha=0.7)

    # # 画纵线
    ax.axvline(x=0,linestyle='--',color='black',linewidth=1 )
    # # ax4.axvline(x=2022,linestyle='--',color='gray')
    # # 垂直填充
    # ax4.axvspan(2007, 2022, facecolor='#e8e8e8', alpha=0.7)
    ax.set_facecolor('none')
    # # 注释
    # ax.annotate(title, (2, 25), fontsize=8)
    # ax4.legend()
    ax.set_ylim(10,80)
    ax.set_yticks(np.arange(10, 91, 20))
    # ax.set_xlim(-5,5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title(title, loc='left', color='black', fontsize=8)

#---------------------- 绘制趋势结果 ------------------------
import matplotlib as mpl
mpl.rcParams['hatch.color'] = 'white'

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


global_t_composite2, cycle_lon = add_cyclic_point(anom_grd, coord=lon)   ## 给数据加上 360度经度 位置的数据
mappable = ax1.contourf(global_t_composite2[:,:],transform=ccrs.PlateCarree(),cmap=cmap_reversed,levels=levels,extend='both') #subplot_kw={'projection':projection,'frameon':False},extend='both')

# 在画图之前设置 hatch 的线宽（决定点的大小）
plt.rcParams['hatch.linewidth'] = 1.2  # 默认是 1.0，调大让点变大

p_t_value = p_values_grd
c1p = ax1.contourf( p_t_value[:,:], levels =[0,0.1,1],hatches=['....',None], colors=[(1,1,1,0),(1,1,1,0)],transform=ccrs.PlateCarree())

for collection in c1p.collections:
    collection.set_edgecolor('none')
    collection.set_linewidth(0)
    collection.set_rasterized(True)




ax1.axis('off') 
new_position = fig.add_axes([1.08, 0.65,0.39, 0.015])   
ax1.add_feature(cfeature.COASTLINE,  linewidth=0.5 , edgecolor='black')
ax1.add_feature(cfeature.BORDERS, linestyle = 'dotted',lw = 0.35, edgecolor='black')

cbar = plt.colorbar(mappable,cmap=cmap_reversed,norm=norm,  cax=new_position,**cbar_kwargs)
cbar.outline.set_color('none')
cbar.update_ticks()
cbar.ax.xaxis.set_tick_params(labelsize=9)
ax1.set_title(' Trends of T$_{cir}$ in JA',loc='center',color='black',fontsize=13,y=1.03)
ax1.text(0.06,1.1, 'a', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax1.transAxes)

ax1.set_xlabel(' ')
ax1.set_ylabel(' ')
ax1.set_aspect(1.5)



### ============== Fig.b ======================



MMM_T_anom = mask_land(MMM_T_anom,'ocean','lon')
global_t_composite3, cycle_lon = add_cyclic_point(MMM_T_anom, coord=lon)   ## 给数据加上 360度经度 位置的数据
mappable = ax2.contourf(global_t_composite3[:,:],transform=ccrs.PlateCarree(),cmap=cmap_reversed,levels=levels,extend='both') #subplot_kw={'projection':projection,'frameon':False},extend='both')


# p_t_value = p_t_test_pos
# c2p = ax2.contourf( p_t_value[:,:], levels =[0,13.5,20],hatches=[ None,'....'],colors="none", transform=ccrs.PlateCarree())
# agreement_plot = np.where(Agree_ratio, 1, np.nan)
c2p = ax2.contourf(Agree_num, levels=[0,13.5,20],hatches=[None,'....'],colors=[(1,1,1,0),(1,1,1,0)], transform=ccrs.PlateCarree())


for collection in c2p.collections:
    collection.set_edgecolor('white')
    collection.set_linewidth(0)
    collection.set_rasterized(True)
 


# p_t_value = p_t_test_neg
# c2p = ax2.contourf( p_t_value[:,:], levels =[0,13.5,20],hatches=[ None,'....'],colors="none", transform=ccrs.PlateCarree())

# for collection in c2p.collections:
#     collection.set_edgecolor('white')
# for collection in c2p.collections:
#     collection.set_linewidth(0)

ax2.axis('off')

ax2.add_feature(cfeature.COASTLINE,  linewidth=0.5 , edgecolor='black')
ax2.add_feature(cfeature.BORDERS, linestyle = 'dotted',lw = 0.35, edgecolor='black')
ax2.set_title(' Multi-model mean of trends in T$_{cir}$',loc='center',color='black',fontsize=13,y=1.03)
ax2.text(0.06,1.1, 'c', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax2.transAxes)

ax2.text(0.51,0.3, 'stippling: ', fontsize=12.5,  fontfamily='sans-serif',transform=ax2.transAxes)
ax2.text(0.51,0.17, '≥14 models = ERA5 sign', fontsize=12,  fontfamily='sans-serif',transform=ax2.transAxes)


############
# spatial correlation
Spatial_r = spatial_corr(global_t_composite2.data, global_t_composite3.data)
print("figure1 & figure 2 similarity:",Spatial_r)
n =len(global_t_composite2.data.flatten())
p = pearson_significance(Spatial_r,n)
print(f"NA Spatial correlation: {Spatial_r:.2f}")
print(f"Naive p-value: {p:.8f} (可能低估真实p值)")

ax2.set_xlabel(' ')
ax2.set_ylabel(' ')
ax2.set_aspect(1.5)
 
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




###----------------------------------------------------------------
### double box
###----------------------------------------------------------------


data_raw = {"EE":Model_raw_EE_trend,
            "CC":Model_raw_CA_trend,
            "ES":Model_raw_NA_trend,
            "WNA":Model_raw_WNA_trend}; data_raw = pd.DataFrame(data_raw) 

data_dyn = {"EE":Model_dyn_EE,
            "CC":Model_dyn_CA,
            "ES":Model_dyn_NA,
            "WNA":Model_dyn_WNA} ; data_dyn = pd.DataFrame(data_dyn) 

Era5_raw = [T_EE_trend,T_CA_trend,T_NA_trend,T_WNA_trend]

Era5_dyn = [dyn_EE_trend,dyn_CA_trend,dyn_NA_trend,dyn_WNA_trend]

x1 = np.arange(1,12,3) 
x2 = x1+1
# x3 = x2+1
print(Model_raw_WNA_trend)
print(Model_raw_CA_trend)
print(Model_raw_EE_trend)
print(Model_raw_NA_trend)

print(Model_dyn_WNA)
print(Model_dyn_CA)
print(Model_dyn_EE)
print(Model_dyn_NA)

box1 = ax22.boxplot(data_raw.dropna(), widths=0.4,positions=x1, patch_artist=True,showmeans=True,
            boxprops={"facecolor": "#fce2d2",
                      "edgecolor": "#ec9374",
                      "linewidth": 0.5},
            medianprops={"color": "#ec9374", "linewidth": 2},
            meanprops={'marker':'o',
                       'markerfacecolor':'#ec9374',
                       'markeredgecolor':'#ec9374',
                       'markersize':6},
            capprops={'color': "#fce2d2", 'linewidth': 1},
            flierprops={'marker': 'o','markerfacecolor':'none','markeredgecolor':"#fce2d2", 'markersize': 5},
            whiskerprops={'color':"#fce2d2",'linestyle':'--','linewidth':1.2})



box2 = ax22.boxplot(data_dyn.dropna(),widths=0.4,positions=x2,patch_artist=True,showmeans=True,
            boxprops={"facecolor": "#dae9f2",
                      "edgecolor": "#569fc9",
                      "linewidth": 0.5},
            medianprops={"color": "#569fc9", "linewidth": 2},
            meanprops={'marker':'o',
                       'markerfacecolor':'#569fc9',
                       'markeredgecolor':'#569fc9',
                       'markersize':6},
            capprops={'color': "#dae9f2", 'linewidth': 1},
            flierprops={'marker': 'o','markerfacecolor':'none','markeredgecolor':"#dae9f2", 'markersize': 5},
            whiskerprops={'color': "#dae9f2",'linestyle':'--','linewidth':1.2})

box3 = ax22.scatter(x1, Era5_raw, marker='s',s=50, facecolor="none",zorder=10,edgecolors="purple",linewidth=2,label='ERA5 - origin trend')
box4 = ax22.scatter(x2, Era5_dyn, marker='x',s=50, color="blue",zorder=10,label='ERA5 - Dynamic trend')
print("WNA Q1 =", box2['boxes'][3].get_path().vertices[:,1].min(),"          WNA Q3 =", box2['boxes'][3].get_path().vertices[:,1].max(),  "   mean",box2['means'][3].get_ydata()[0])

city = ["EE","NWC","ES","WNA"]
ax22.set_xticks([1.5,4.5,7.5,10.5],city,fontsize=12)
ax22.set_xticklabels(city)  #,rotation=30)
ax22.set_title('Temperature trends',loc='center',color='black',fontsize=13,y=1.01)
ax22.text(-0.115,1.05, 'b', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax22.transAxes)
     
ax22.axhline(y=0 , color='k', linestyle='--',linewidth=1.5,zorder=0)
ax22.set_ylim(-0.2,1.2)
ax22.set_ylabel(' T (°C/decade)',fontsize=13)
ax22.tick_params(axis='both', which='major', labelsize=11)

# 给箱体添加图例，每类箱线图中取第一个颜色块用于代表图例
# ax22 = plt.gca() #返回坐标轴
ax22.legend(handles=[box1['boxes'][0],box2['boxes'][0],box3,box4],labels=['20 model Total','20 model Cir','ERA5 Total','ERA5 Cir'], frameon=False)
# ax22.legend(handles=[box1['boxes'][0],box2['boxes'][0],box3,box4],labels=['20 model Total','20 model Cir','ERA5 Total','ERA5 Cir'], frameon=False)


plt.savefig("/public/master/data1/home/fcai/abc/2Paper_blocking/Final_figures/Fig1_75_HW_trend.pdf", dpi=500, bbox_inches='tight')
plt.show()