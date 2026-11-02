import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings
warnings.filterwarnings("ignore")
import matplotlib as mpl
mpl.use("Agg")


def computing_link_bwt_two_array(array1, array2,years_num,lat,lon,outfile):
    """Computing the links between two metrix.

    Args:
        array1 (bool): climatological data 1;
        array2 (bool): climatological data 2;
        years_num (num): total year number;
        lat (DataArray): latitude information;
        lon (DataArray): longitude information;
        outfile (str): files need to be stored.
    
    Note: here we only calculate from 1979; 
    due to the limited calculation memory, we have to store each year seperately.
    
    """

    for iyear in range(years_num):
        network=np.zeros((np.shape(lat)[0],np.shape(lon)[0],np.shape(lat)[0],np.shape(lon)[0]),dtype=np.float32)
        print(" iyear = " + str(iyear))
        Metrix1=array1[iyear,:,:,:]
        Metrix2=array2[iyear,:,:,:]
        
        
        #确定一个点
        for ilon in range(np.shape(lon)[0]):
            print(" ilon = " + str(ilon))
            for ilat in range(np.shape(lat)[0]): #固定一点看与全球的关系
                print(" ilat = " + str(ilat))
                # if np.isnan(Metrix1[:,:,ilat,ilon]).all():
                #     break
                # else:
                Metrix1_3D=np.tile( Metrix1[:,ilat,ilon],(np.shape(lat)[0],np.shape(lon)[0],1)).transpose(2,0,1)
                concurrent=np.sum(Metrix2[:,:,:]+ Metrix1_3D[:,:,:]==2, axis=0)
                network[ilat,ilon,:,:] = concurrent 
        
        ##--##--##-- 存储 --##--##--##
        network = xr.DataArray(data= network, dims=['lat1','lon1','lat2','lon2'],
                    coords= {'lat1':lat.data , 'lon1':lon.data, 'lat2':lat.data , 'lon2':lon.data})

        ds3 = xr.Dataset(data_vars= dict(network = network))
        ds3.to_netcdf(outfile+f"concurrent_day_"+str(iyear+1979)+".nc")
        ds3.close()
        del network


def significant_links_at_each_grid(array1, array2,lat,lon,Test_file,outfile):
    """We have a threshold sheet, now we use this table to calculate the threshold for each grid cell.
    
    Args:
        array1 (bool): climatological data 1;
        array2 (bool): climatological data 2;
        lat (DataArray): latitude information;
        lon (DataArray): longitude information;
        outfile (str): files need to be stored.
    
    """
    ## ------------- 1. total days for each grid point
    # 每个点的热浪天数
    array1_days =  np.sum(array1[:,:,:,:], axis=(0,1))
    array1_days = np.where(array1_days>=1100, 1100,array1_days)
    print(int(np.nanmax(array1_days)))

    # 每个点的急流天数
    array2_days = np.sum(array2[:,:,:,:], axis=(0,1))
    array2_days = np.where(array2_days>=1100, 1100,array2_days)
    print(int(np.nanmax(array2_days)))

    ## -------------  2. 创建 network 数据  

    threshold_values = np.zeros((np.shape(lat)[0],np.shape(lon)[0],np.shape(lat)[0],np.shape(lon)[0]))

    # load threshold sheet
    TEST = xr.open_dataset(Test_file)
    matrix995 = TEST.matrix995

    # 把表格变成 （ilat1,ilon1, ilat2,ilon2）
    for ilat in range(np.shape(lat)[0]):
        print(f"ilat{ilat}")
        for ilon in range(np.shape(lon)[0]):
            print(f"ilon{ilon}")
            

            # 获取热浪事件的天数
            hw_event_days = int(array1_days[ilat, ilon])
            
            for b_days in np.unique(array2_days.astype(int)):
                concurrent_days = matrix995[hw_event_days,b_days]
                threshold_values[ilat,ilon,:,:][np.where(array2_days==b_days)] =concurrent_days
                del concurrent_days
            del hw_event_days
            
    # 确定一个点热浪天数，让西风天数成为一个阈值表 matrix(热浪set， 对应0-700), 然后 各点西风为westerly days = (westerly_true == 0-700)

    ##--##--##--##--  同步网络矩阵, 存入nc文件  --##--##--##--##
    matrix4D_995_array0 = xr.DataArray(data=threshold_values, dims=['lat1', 'lon1', 'lat2', 'lon2'],
                                    coords={'lat1':lat.data,'lon1':lon.data,
                                            'lat2':lat.data,'lon2':lon.data})
    ds0 = xr.Dataset(data_vars=dict(matrix4D_995=matrix4D_995_array0))
    ds0.to_netcdf(outfile)
    ds0.close()




def get_significant_links(infile,significant_point_eachgrid_file,lat,lon,years_num,outfile):
    """
    Construct a conservative thresholded network.

    Step 1:
        Retain only the top 5% strongest links based on total concurrent days.

    Step 2:
        Among these top 5% links, further retain only links exceeding the
        pairwise event-count-based randomization threshold.


    Args:

        infile (str): files storing the links between two arrays;
        significant_point_eachgrid_file (str): Threshold table for each grid point
        lat (DataArray): latitude information;
        lon (DataArray): longitude information;
        years_num (num): total year number;
        outfile (str): files need to be stored.
    
    """

    ##--##--##--##--  创建 network 数据  --##--##--##--##
    concurrent_day=np.zeros((years_num,np.shape(lat)[0],np.shape(lon)[0],np.shape(lat)[0],np.shape(lon)[0]))
    for iyear in range(years_num):
        print('iyear = ', iyear + 1979)
        ds0 = xr.open_dataset(infile+f"/concurrent_day_{iyear+1979}.nc")
        # 调整经度坐标
        for lon_name in ['lon1', 'lon2']:
            ds0['longitude_adjusted'] = xr.where(ds0[lon_name] > 180, ds0[lon_name] - 360, ds0[lon_name])
            ds0 = ds0.swap_dims({lon_name: 'longitude_adjusted'}).sel(
                **{'longitude_adjusted': sorted(ds0.longitude_adjusted)}
            ).drop(lon_name)
            ds0 = ds0.rename({'longitude_adjusted': lon_name})
        
        network1  = ds0.network
        lat = ds0.lat1; lon = ds0.lon1
        concurrent_day[iyear,:,:,:,:]= network1  

    
    All_year_concurr= np.sum(concurrent_day,axis=0)
    print(np.shape(All_year_concurr))

    # #Step1: 只分析最强的top5%
    # matrix4D_90_step1 = np.nanpercentile(All_year_concurr, 90)
    # All_year_concurr1  = np.where(All_year_concurr < matrix4D_90_step1, 0, All_year_concurr )      #[:,:,:,:]
 


    # Step2: 都过前面那个弱显著性检验的99th 
    # 加载测试数据集和阈值表
    TEST = xr.open_dataset(significant_point_eachgrid_file)
    for lon_name in ['lon1', 'lon2']:
        TEST['longitude_adjusted'] = xr.where(TEST[lon_name] > 180, TEST[lon_name] - 360, TEST[lon_name])
        TEST = TEST.swap_dims({lon_name: 'longitude_adjusted'}).sel(
            **{'longitude_adjusted': sorted(TEST.longitude_adjusted)}
        ).drop(lon_name)
        TEST = TEST.rename({'longitude_adjusted': lon_name})

    matrix995 = TEST.matrix4D_995
    All_year_concurr2 = np.where(All_year_concurr < matrix995, 0, All_year_concurr )    #[:,:,:,:]

    matrix4D_995_array0 = xr.DataArray(data=All_year_concurr2, dims=['lat1', 'lon1', 'lat2', 'lon2'],
                                coords={'lat1':lat.data,'lon1':lon.data,
                                        'lat2':lat.data,'lon2':lon.data})
    ds1 = xr.Dataset(data_vars=dict(significant_99=matrix4D_995_array0))
    ds1.to_netcdf(outfile)
    ds1.close()
