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
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
 
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


## ------------- 3. 对应 U 风 average change

ds4 = xr.open_dataset("/public/data2/model/fcai/data-observation/ERA5_daily/uwind/uwind.1979-01.daily.nc")
print(ds4)
u = ds4.u.loc[:,300,80:0,:][:,::-1,:]
lat = u.latitude; lon = u.longitude;print(lat) 
print(np.shape(u))
u250 = np.zeros((45,62,np.shape(u)[1],np.shape(u)[2]))

for iyear in range(45):
  print('iyear = ',iyear+1979)

  zJuly = xr.open_dataset("/public/data2/model/fcai/data-observation/ERA5_daily/uwind/uwind."+str(iyear+1979)+"-07.daily.nc")
  u250[iyear,0:31,:,:] = zJuly.u.loc[:,300,80:0,:][:,::-1,:]
  zAugust = xr.open_dataset("/public/data2/model/fcai/data-observation/ERA5_daily/uwind/uwind."+str(iyear+1979)+"-08.daily.nc")
  u250[iyear,31:62,:,:] = zAugust.u.loc[:,300,80:0,:][:,::-1,:]
  zJuly.close(); zAugust.close()

ua_year_run_90 = np.mean(u250[:,:,:,:],1)
u_JJA_clm = np.mean(ua_year_run_90[2:32,:,:],0)
tile_u_JJA_clm =np.tile(u_JJA_clm ,(45,1,1))
u_ano = np.subtract(ua_year_run_90[:,:,:],tile_u_JJA_clm)


##--##--## 计算趋势 --##--##--##
nt = 45;nlat=81;nlon=360
U_result=trend_result(u_ano,nt,nlat,nlon)
u_p_values_grd = U_result[1]
u_anom_grd = U_result[0]

##--##--##--##-- Fig cde  --##--##--##--##
# Westerly
ds0=xr.open_dataset("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/2_U300positive_network/PDF/0_30_PDF_U300_positive.nc")
pdf_0_30 =ds0.pdf
pdf_0_30 = pdf_0_30/np.sum(pdf_0_30)

ds1=xr.open_dataset("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/2_U300positive_network/PDF/30_60_PDF_U300_positive.nc")
pdf_30_60 =ds1.pdf
pdf_30_60 = pdf_30_60/np.sum(pdf_30_60)

ds2=xr.open_dataset("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/2_U300positive_network/PDF/60_90_PDF_U300_positive.nc")
pdf_60_90 =ds2.pdf
pdf_60_90 = pdf_60_90/np.sum(pdf_60_90)


# Easterly
ds00=xr.open_dataset("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/3_U300negative_network/PDF/0_30_PDF_U300_negative.nc")
Easterly_pdf_0_30 =ds00.pdf
Easterly_pdf_0_30 = Easterly_pdf_0_30/np.sum(Easterly_pdf_0_30)

ds11=xr.open_dataset("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/3_U300negative_network/PDF/30_60_PDF_U300_negative.nc")
Easterly_pdf_30_60 =ds11.pdf
Easterly_pdf_30_60 = Easterly_pdf_30_60/np.sum(Easterly_pdf_30_60)

ds22=xr.open_dataset("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/3_U300negative_network/PDF/60_90_PDF_U300_negative.nc")
Easterly_pdf_60_90 =ds22.pdf
Easterly_pdf_60_90 = Easterly_pdf_60_90/np.sum(Easterly_pdf_60_90)




##--##--##--##-- Fig f --##--##--##--##
# ------------------------------------------------------------ # #
# --------------------    read networks    ------------------- # #
ds0=xr.open_dataset("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/2_U300positive_network/PDF_U300_postive.nc")
pdf_westerly=ds0.pdf
# print(np.max(pdf_westerly))
# print(np.argmax(pdf_westerly))

ds1 = xr.open_dataset("/public/home/fcai/abc/2Paper_jet/Fig2/Fig2_new/3_U300negative_network/PDF_U300_negative.nc")
pdf_easterly=ds1.pdf.values
# print(np.max(pdf_easterly))
# print(pdf_easterly)
# exit()

#----------------------分割线（画图函数）------------------------
with open('/public/home/fcai/abc/data/CN-border-La.dat') as src:
    context = src.read()
    blocks = [cnt for cnt in context.split('>') if len(cnt) > 0]
    borders = [np.fromstring(block, dtype=float, sep=' ') for block in blocks]


# Create a figure and axis
fig = plt.figure(figsize=[12, 12],frameon=True)
# ax1 = fig.add_axes([0.0, 0.7, 0.8, 0.25], projection=ccrs.PlateCarree() )
# ax2 = fig.add_axes([0.0, 0.53, 0.8, 0.25], projection=ccrs.PlateCarree() )

# ax33 = fig.add_axes([0, 0.45, 0.2,0.072])
# ax44 = fig.add_axes([0.265, 0.45, 0.2,0.072])
# ax55 = fig.add_axes([0.53, 0.45, 0.2, 0.072])
projection = ccrs.Robinson(central_longitude=150)
# projection = ccrs.PlateCarree(central_longitude=150)
ax1 = fig.add_axes([0.08, 0.92, 0.8, 0.4], projection=projection)
ax33 = fig.add_axes([0.08, 0.78, 0.25, 0.125])
ax44 = fig.add_axes([0.40, 0.78, 0.25, 0.125])
ax55 = fig.add_axes([0.72, 0.78, 0.25, 0.125])

ax66 = fig.add_axes([0.18, 0.385, 0.65, 0.20])



# prepare 
box = [-180, 180, 10, 80]
scale = '50m'            
xstep, ystep = 90, 35




def make_composite(composite,lat,lon,Colors,ax,title,levels):

    #设置色标的
    cbar_kwargs = {
        'orientation': 'vertical',  # 改为水平方向
        'label': '',
        'ticks':[50,60,70,80,90],
        'pad': 0.03,
        'shrink': 0.38,  #长度
        'extend': 'both',
        'aspect': 10,    # 宽度
        'fraction':0.1
    }

    #设置画数据的精度，这里设置了从-30到30度，按1度画温度
    levels = levels
    Colors=Colors
   
    # global_t_composite2, cycle_lon = add_cyclic_point(composite, coord=lon)   ## 给数据加上 360度经度 位置的数据
    c=ax.contourf(lon,lat ,composite, cbar_kwargs=cbar_kwargs,transform=ccrs.PlateCarree(),colors=Colors,levels=levels,extend='both')#levels=levels,
    # new_position = fig.add_axes([0.73, 0.75,0.017, 0.15])   

    cb= fig.colorbar(c,  **cbar_kwargs  )#cax=new_position,
    cb.ax.xaxis.set_tick_params(labelsize=15)
    ax.set_title(title,loc='Center',color='black',fontsize=20,y=1.1)
    ax.set_xlabel(' ')
    ax.set_ylabel(' ')
    ax.set_aspect(1.0)
    return ax


#x,y及Z数据信息
lat_interval = np.linspace(-40, 40, 21); print(lat_interval)
lon_interval = np.linspace(-80, 80, 41); print(lon_interval)
 
def contour_2D(data,ax):
    levels = [3,5,7] 
    ax.contour(lon_interval,lat_interval,data,levels=levels,  shading='None')
  
 
def make_2D(data,ax,title,levels,Colors):
    # levels = [0,0.0025,0.005,0.0075, 0.01,0.0125,0.015,0.0175,0.02,0.025]
    levels = levels 
    Colors=Colors
    cmap = LinearSegmentedColormap.from_list('custom_cmap', Colors)
    
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    c = ax.pcolormesh(lon_interval,lat_interval,  data , cmap=cmap,norm=norm, shading='auto')#, vmin=0, vmax=0.02

    # 设置标题和其他参数
    ax.set_title(title, fontsize=13,loc='Center')
    ax.axvline(0,linestyle='--',color='black',linewidth=1 )
    ax.axhline(0,linestyle='--',color='black',linewidth=1 )
    ax.set_xlim(-50,51)
    ax.set_ylim(-30,31)
    ax.tick_params(axis='both', labelsize=12)

 
### ============== Fig.1 ======================
# 在画图之前设置 hatch 的线宽（决定点的大小）
import matplotlib as mpl
mpl.rcParams['hatch.color'] = 'white'

cmap = plt.get_cmap('PuOr')
cmap_colors = cmap(np.arange(cmap.N))
cmap_reversed = mcolors.LinearSegmentedColormap.from_list('Reversed', cmap_colors[::-1])

#设置色标的
cbar_kwargs = {
    'orientation': 'vertical',  # 改为水平方向
    'label': 'm/s/decade',
    'ticks': np.arange(-1.6, 1.6 + 0.8, 0.8) ,
    'pad': 0.03,
    'shrink': 0.55,
    'extend': 'both',
    'extendfrac': 0.1  
}

#设置画数据的精度，这里设置了从-30到30度，按1度画温度
levels = np.arange(-1.6, 1.6 + 0.2, 0.2)

global_t_composite2, cycle_lon = add_cyclic_point(u_anom_grd, coord=lon)   ## 给数据加上 360度经度 位置的数据
mappable2 = ax1.contourf(global_t_composite2,transform=ccrs.PlateCarree(),cmap=cmap_reversed,levels=levels,extend='both') #subplot_kw={'projection':projection,'frameon':False},extend='both')
ax1.axis('off') 

# 在画图之前设置 hatch 的线宽（决定点的大小）
plt.rcParams['hatch.linewidth'] = 1.2  # 默认是 1.0，调大让点变大
c2p = ax1.contourf( u_p_values_grd, levels =[0,0.1,1],hatches=['....',None],colors="none", transform=ccrs.PlateCarree())

for collection in c2p.collections:
    collection.set_edgecolor('white')
for collection in c2p.collections:
    collection.set_linewidth(0)
 
new_position = fig.add_axes([0.9, 1.02,0.015, 0.2])   
ax1.add_feature(cfeature.COASTLINE,  linewidth=0.5 , edgecolor='black')
ax1.add_feature(cfeature.BORDERS, linestyle = 'dotted',lw = 0.25, edgecolor='black')

cbar = plt.colorbar(mappable2,norm=norm,  cax=new_position,**cbar_kwargs)
cbar.outline.set_color('none')
cbar.update_ticks()
cbar.ax.xaxis.set_tick_params(labelsize=9)
ax1.set_title(' Trends of U300 in JA',loc='Center',color='black',fontsize=13,y=1.03)
ax1.text(0.06,1.14,'a', fontsize=15.5,fontweight='bold', fontfamily='sans-serif',transform=ax1.transAxes)
# ax1.set_title('JA',loc='right',color='black',fontsize=13,y=1.03)
ax1.set_xlabel(' ')
ax1.set_ylabel(' ')
ax1.set_aspect(1.5)


# 画曲线
lat1 = 57
lon1 = -130
lat2 = 32
lon2 = -88

npts = 100
lons = np.zeros((npts))
lats = np.zeros((npts))
for i in range(npts):
    t = i / (npts - 1.0)
    lons[i] = lon1 + t * (lon2 - lon1)
    lats[i] = lat1 + t * (lat2 - lat1)
    lats[i] = lats[i] - 7.0 * np.sin(3.14159 * t)
    lons[i] = lons[i]- 10.0 * np.sin(3.14159 * t)

 
# 绘制大圆线
ax1.plot(lons, lats, 
       color='red',
       linewidth=3,
       transform=ccrs.Geodetic())


k = 1
#East Europe
locals()['rectangle' + str(k)] = Rectangle(
    (59, 32),
    0.1, 39, 
    linewidth=3, edgecolor='red',
    facecolor='none',linestyle='-',
    transform=ccrs.PlateCarree()
)
locals()['ax' + str(k)].add_patch(locals()['rectangle' + str(k)])


#Central Asia
locals()['rectangle' + str(k)] = Rectangle(
    (105, 15),  # 左下角坐标 (longitude, latitude)
    0.1, 30, # 宽度和高度 (longitude, latitude)
    linewidth=3,edgecolor='red',
    facecolor='none',linestyle='-',
    transform=ccrs.PlateCarree()
)
locals()['ax' + str(k)].add_patch(locals()['rectangle' + str(k)])  

#North Asia
locals()['rectangle' + str(k)] = Rectangle(
    (155, 43), 
    0.1, 30, # 宽度和高度 (longitude, latitude)
    linewidth=3,edgecolor='red',
    facecolor='none',linestyle='-',
    transform=ccrs.PlateCarree()
)
locals()['ax' + str(k)].add_patch(locals()['rectangle' + str(k)]) 






### ============== Fig.2 ======================
# EE
levels =[0, 1.2,2,3,4,6,8,10]
Colors_EE= ('#ffffff','#deebf7','#bce5f8','#8acbef',\
            '#4a91cc','#1f4182')      #  ('#4999a1','#4c8ec5','#80bee5','#c0e6fd','#ffffff')
# Colors_EE=Colors_EE[:][::-1]
new_Colors_EE =[]
for item in Colors_EE:
    rgba = mcolors.to_rgba(item)
    if item == '#ffffff':
        new_Colors_EE.append((rgba[0],rgba[1],rgba[2],0))
    else:
        new_Colors_EE.append(rgba)


make_2D(Easterly_pdf_0_30*1000 , ax33, "0-30 N", levels,new_Colors_EE); ax33.set_ylabel("lat diff",fontsize=12) #; ax44.set_ylabel("lat diff",fontsize=12)
make_2D(Easterly_pdf_30_60*1000 , ax44, "30-60 N",levels,new_Colors_EE);ax55.set_xlabel("lon diff",fontsize=12) #; ax55.set_ylabel("lat diff",fontsize=12)
ax33.set_xlabel("lon diff",fontsize=12)
ax44.set_xlabel("lon diff",fontsize=12)

cmap = LinearSegmentedColormap.from_list('custom_cmap', new_Colors_EE)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

c1 = ax55.pcolormesh(lon_interval,lat_interval, Easterly_pdf_60_90*1000, cmap=cmap,norm=norm, shading='auto')#, vmin=0, vmax=0.02
ax55.set_title("60-90 N", fontsize=20)
# 配置颜色条
new_position = fig.add_axes([0.41, 0.705,0.23, 0.01]) 
cbar1 = plt.colorbar(c1, ax=ax55, orientation='horizontal' ,cax=new_position,location='bottom' ,pad= 0.18,shrink=0.8)
cbar1.set_label(" ")
cbar1.ax.tick_params(labelsize=0 )


# 设置标题和其他参数

     

ax55.axvline(0,linestyle='--',color='black',linewidth=1 )
ax55.axhline(0,linestyle='--',color='black',linewidth=1 )
ax55.set_xlim(-50,51)
ax55.set_ylim(-30,31)
ax55.tick_params(axis='both', labelsize=12)






# WE

Colors_WE= ('#ffffff','#ffe101','#feaa00',\
            '#ff8000','#ff3234','#da0003') 
new_Colors_WE =[]
for item in Colors_WE:
    rgba = mcolors.to_rgba(item)
    if item == '#ffffff':
        new_Colors_WE.append((rgba[0],rgba[1],rgba[2],0))
    else:
        new_Colors_WE.append(rgba)

#
make_2D(pdf_0_30*1000 , ax33, "0-30 N", levels,new_Colors_WE)
make_2D(pdf_30_60*1000 , ax44, "30-60 N",levels,new_Colors_WE) 


cmap = LinearSegmentedColormap.from_list('custom_cmap', new_Colors_WE)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

c = ax55.pcolormesh(lon_interval,lat_interval, pdf_60_90*1000 , cmap=cmap,norm=norm, shading='auto')#, vmin=0, vmax=0.02

# 配置颜色条
new_position = fig.add_axes([0.41, 0.695,0.23, 0.01]) 
cbar = plt.colorbar(c, ax=ax55, orientation='horizontal' ,cax=new_position,location='bottom' ,pad= 0.18,shrink=0.8)
cbar.ax.tick_params(labelsize=11)  #设置色标刻度字体大小
cbar.set_label("probability * 1000", fontsize=11)
ax55.set_title("60-90 N", fontsize=13)
ax55.set_xlim(-50,51)
ax55.set_ylim(-30,31)

ax33.text(0.0,1.1, 'b', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax33.transAxes)
ax44.text(0.0,1.1, 'c', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax44.transAxes)
ax55.text(0.0,1.1, 'd', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax55.transAxes)


# # ---------------------------- calculate pdf ---------------------------- # #
x = np.zeros((89))
x_axis_data =range(89)
for i in range(89):
    x[i] = (x_axis_data[i] -44 )*2
    # print(x[i])

    # print(pdf_easterly[i])  
    # print(pdf_westerly[i])

# 绘制折线图并添加上下误差范围
ax66.plot( x,pdf_easterly, linestyle='solid', color='black', linewidth=0.5,
             marker='o', markeredgecolor='black', markerfacecolor='#8acbef',
             markeredgewidth=0.3, label='hw & U300-') 
ax66.plot( x,pdf_westerly, linestyle='solid', color='black', linewidth=0.5,
             marker='o', markeredgecolor='black', markerfacecolor='#ff8000',
             markeredgewidth=0.3, label='hw & U300+')  # 替换 'Your Label' 为实际标签

# print( np.argmax(pdf_westerly))
ax66.spines['right'].set_visible(False)
ax66.spines['top'].set_visible(False)
# ax66.spines['bottom'].set_position(('data',-50))
   
    
ax66.set_ylim(0,0.06)


ax66.set_xlim(-50,50)
ax66.text(0.0,1.05, 'e', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax66.transAxes)
ax66.set_title('Tripole wind pattern in the Northern Hemisphere' ,loc='Center', color='black', fontsize=14,x= 0.5,y=1.03)
ax66.tick_params(axis='x',labeltop=False,labelbottom=True, labelsize=12,length = 3, width=0.49) #, labelrotation=45)
ax66.tick_params(axis='y', labelsize=12,length = 3,direction = 'in', width=0.49)
ax66.set_yticks(np.arange(0,0.061,0.01))
ax66.set_xticks(np.arange(-50,60,10))
# ax66.text(0.0,-0.7, 'e', fontsize=15.5, fontweight='bold', fontfamily='sans-serif',transform=ax66.transAxes)

# ax66.yaxis.tick_right()
# 画横线
# ax66.axhspan(lat1, lat2, facecolor='#e8e8e8', alpha=0.7)

# 画纵线
ax66.axvspan(-1, 1, facecolor='#e8e8e8', alpha=0.7)

ax66.axvline(12, 0,0.93, color='#e0311f', linestyle='--', linewidth=1)
ax66.axvline(-18,0, 0.335, color='#e0311f', linestyle='--', linewidth=1)
ax66.axvline(-4,0, 0.87, color='#7caff3', linestyle='--', linewidth=1)


# 显示图例
ax66.legend(fontsize=10,frameon=False, bbox_to_anchor=(0.93,1))

# plt.xlabel('lat difference (°N)', labelpad=-40)
ax66.set_xlabel('lat difference (°N)', labelpad=12,fontsize=12)
ax66.set_ylabel('Probability', labelpad=12,fontsize=12)

# 一点批注
ax66.text(0.27,0.10, '-18°', color='#e0311f',fontsize=12,transform=ax66.transAxes)
ax66.text(0.423,0.6, '-4°',color='#7caff3', fontsize=12,transform=ax66.transAxes)
ax66.text(0.56,0.6, '+12°',color='#e0311f', fontsize=12,transform=ax66.transAxes)


 
# #Central Asia
# for k in range(1,3):


#     #Central Asia
#     locals()['rectangle' + str(k)] = Rectangle(
#         (18, 45),
#         40, 20, 
#         linewidth=1.8,edgecolor='red',
#         facecolor='none',
#         transform=ccrs.PlateCarree()
#     )
#     locals()['ax' + str(k)].add_patch(locals()['rectangle' + str(k)]) 
  

plt.savefig("/public/home/fcai/abc/2Paper_blocking/Final_figures/Fig2_Network.png", dpi=800, bbox_inches='tight')     #
 