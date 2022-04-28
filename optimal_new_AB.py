
import pandas as pd
import numpy as np
from numpy import exp, log

from scipy.stats import zscore, skew, kurtosis
from sklearn.preprocessing import  Normalizer, StandardScaler, MinMaxScaler
# loading essentials libraries

import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
import seaborn as sns
from collections import defaultdict, OrderedDict
import random
from sklearn.metrics import silhouette_samples, silhouette_score, euclidean_distances

import matplotlib.cm as cm

from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.patheffects import AbstractPathEffect

from numpy import median, std
import os

from scipy.spatial import ConvexHull

#import matplotlib.colors as cm
import matplotlib.colors as mcolors
from sklearn.metrics import pairwise
from scipy.spatial import distance
from scipy import interpolate
from sklearn.metrics import mean_squared_error

import nolds
import kneed

from numpy import abs, mean, sum, ceil
from scipy.spatial import ConvexHull




class optimal_sensor ():
    '''
    Non-keywords arguments
    temp, humid, 
    '''
    def __init__(self, data_path, data_locs_path,  partitions, clus_num, sensor_ids, sections, params):
        self.path = data_path
        self.path_locs = data_locs_path
        self.params = params
        self.sections = sections
        self.partitions = partitions
        self.clus_num = clus_num
        self.sensor_ids = sensor_ids
        self.data_locs = pd.read_csv(self.path_locs)
        # convert string to numerical


            
    def data_converter(self,data):
        '''
        functon that converts each columns of the data frame to numeric
        '''
        
        for col_ind in range(len(data.columns)):
            data.iloc[:,col_ind] = pd.to_numeric(data.iloc[:,col_ind], errors='ignore')

        return data 
    
    # dealing with missing values
    def data_missing_values(self,data):
        '''
        functon that fills missing values in data
        '''
        # data.fillna(data.median(), inplace = True)
        data.dropna(inplace = True)
        

        return data
    
    def data_outliers(self,data):
        ''''
        function that removes outliers
        using interquartile 
        '''
        q1 = data.quantile(0.25) # 25 percentile
        q3 = data.quantile(0.75) # 75 percentile
        iqr = q3 -  q1 # interquartile range
        lower = q1 -  (1.5 *  iqr) #lower limit
        upper = q3 + (1.5 * iqr)  #upper limit
        data = data[( data > lower) | (data< upper)]
        return data

    def feature_engineering(self,data):
        #converting the dataframe to array for proper handling
        
        df = data.values
        ss = StandardScaler() #standard scalar
        df = ss.fit_transform(df)
        nz = Normalizer()  
        df = nz.fit_transform(df)

        if len(df) == len(self.sensor_ids):
            df = pd.DataFrame(df, index=  self.sensor_ids)
        else: 
            df = pd.DataFrame(df)
        return df
    
        
    def data_loading(self):
        '''
        This funtion will check data type and load 
        '''
        

        # cloading of csv file
        try:
            df = pd.read_csv(self.path, encoding= 'unicode_escape') #loading the data
            df = df.iloc[:, 1:]
            df = pd.DataFrame(df)
            print ('File is CSV')
        
        #loading of excel file
        except:
            df = pd.read_excel(self.path, ) #loading the data
            df = df.iloc[:, 1:]
            df = pd.DataFrame(df)
            
            print ('File is excel')
        return df
    

    def pyschometric_tranformation(self,data):
    
        # temperature
        data_temp = data.iloc[:,::2]
        data_temp = pd.DataFrame(data_temp.values, columns=self.sensor_ids)
        # humidity 
        data_humid = data.iloc[:,1::2]
        data_humid = pd.DataFrame(data_humid.values, columns=self.sensor_ids)

        RA = 0.287
        P = 101325

        data_dew = pd.DataFrame()  #dew point temperature
        data_spv = pd.DataFrame()  # specific olume
        data_enth  = pd.DataFrame()#enthalpy
        data_hur = pd.DataFrame() #humid ratio
        Pvs = pd.DataFrame()
        Pv = pd.DataFrame()
        for i , j , z,   in zip (range(0,len(data.columns), 2), range(1,len(data.columns),2), range(len(self.sensor_ids)) ):
            # 𝑃𝑣𝑠  #saturated vapour pressure
            # 𝑃𝑣  #partial vapour pressure
            data_dew[self.sensor_ids[z]] = (data.iloc[:,i]) - ((100 - (data.iloc[:, j]))/5)
            𝑃𝑣𝑠[self.sensor_ids[z]] = (exp(31.9602 - (6270.3605/(data.iloc[: ,i]+273))- (0.46057 * log(data.iloc[:,i]+273)))) #not need
            𝑃𝑣[self.sensor_ids[z]] = ((data.iloc[:, j]) * 𝑃𝑣𝑠.iloc[:,z]) #not need
            data_hur[self.sensor_ids[z]] = (0.622 * 𝑃𝑣.iloc[:,z])/(P - 𝑃𝑣𝑠.iloc[:,z])
            data_enth[self.sensor_ids[z]] = ((1006*(data.iloc[:,i])) + (data_hur.iloc[:,z]*(2515131.0 + (1552.4* (data.iloc[:,i])))))
            data_spv[self.sensor_ids[z]] = (RA*(data.iloc[:,i] +273))/ (P - Pv.iloc[:,z])
        
       
        return data_temp, data_humid, data_dew, data_hur, data_enth, data_spv

    def data_preprocessing(self, data): 
        df = data 
        # conversion to numeric if the data is stored as as numerical string
        df = self.data_converter(df)

        # missing values fill
        df = self.data_missing_values (df)
        #removing outliers 
        df = self.data_outliers(df)

        return df
    
    

    def applied_feature_extraction(self, data):
        '''
        Function that extracts 6 features based on partitions
        mean, median, std, skew, kurtorsis and lypanuov exp
        '''

        df = pd.DataFrame(index=self.sensor_ids) #dataframe of features extraction 
        df_x = data.T.values # data values

        for ind, arr in zip (range(self.partitions), np.array_split(df_x, self.partitions, axis= 1)):
            l_exp_list = [] # list to store lypunov exponential 

            # looping through each sensors
            for j in range(56):
                l_exp  = nolds.lyap_r(arr[j,:], fit='poly')
                l_exp_list.append(l_exp)  

            df[str(str(ind+1)+'_mean')] = mean(arr, axis= 1) #mean
            df[str(str(ind+1)+'_median')] = median(arr, axis= 1) #median
            df[str(str(ind+1)+'_std')] = std(arr, axis= 1) #std
            df[str(str(ind+1)+'_skew')] = skew(arr, axis= 1) #skew
            df[str(str(ind+1)+'_kurt')] = kurtosis(arr, axis= 1) #kurtosis
            df[str(str(ind+1)+'_lyexp')] = l_exp_list #lypunaov exp
            # df[str(str(ind+1)+'_lyexp')] = kurtosis(arr, axis= 1) #lypunaov exp
        
        return df
    def dynamic_plot (self,data,name):

        '''
        Function that describes the dynamic state of each features
        '''
        fig = plt.figure(figsize=(25,8),facecolor= None, edgecolor='black', frameon=True)
        axs  = fig.subplots(2,3,)
        fig.subplots_adjust(hspace=0.3, wspace= 0.3)

        #plt.gca().set_size_inches(20,30)

        plt.rcParams['font.size'] = 12
        feature_list = ['Mean', 'Median', 'Standard deviation', 'Skewness', 'Kurtosis','Lyapunov Exp']
        # feature_list = ['Mean', 'Median', 'Standard deviation', 'Skewness', 'Kurtosis']


        for i in range (2):
            for j in range(3):
                
                if i == 0:

                    marker = '.'
                    mean_f = data.iloc[:, j::6].mean()
                    std_f = data.iloc[:, j::6].std()
                    #min_f = data.iloc[:, j::6].min()
                    #max_f = data.iloc[:, j::6].max()

                    x_span = list(range(1,self.partitions+1)) #numbers of weeks

                    axs[i,j].plot(x_span , (mean_f - 3*std_f), label= 'lower bound', linestyle='dashed', color ='green')
                    axs[i,j].plot(x_span , mean_f, label= 'Feature_avg' , color ='blue', marker= marker)
                    axs[i,j].plot(x_span , (mean_f+ 3*std_f), label= 'upper bound', linestyle='dashed', color = 'green')
                    axs[i,j].fill_between(x_span, (mean_f- 3*std_f), (mean_f+ 3*std_f), fc='tab:green', alpha=0.2)
                    axs[i,j].set_title(feature_list[j])
                    #axs[i,j].set_ylim(min_f, max_f)
                    axs[i,j].set_xlabel("Numbers of weeks")
                    axs[i,j].set_xticks(x_span)
                    axs[i,j].grid()
                elif  i == 1:
                    marker = '.'
                    mean_f = data.iloc[:, j+3::6].mean()
                    std_f = data.iloc[:, j+3::6].std()
                    #min_f = data.iloc[:, j+3::6].min()
                    #maxs_f = data.iloc[:, j+3::6].max()

                    x_span = list(range(1, self.partitions+1))

                    axs[i,j].plot(x_span , (mean_f - 3*std_f), label= 'lower bound', linestyle='dashed', color ='green')
                    axs[i,j].plot(x_span , mean_f, label= 'Feature_avg' , color ='blue', marker= marker)
                    axs[i,j].plot(x_span , (mean_f+ 3*std_f), label= 'upper bound', linestyle='dashed', color = 'green')
                    axs[i,j].fill_between(x_span, (mean_f- 3*std_f), (mean_f+ 3*std_f), fc='tab:green', alpha=0.2)
                    #axs[i,j].set_ylim(min_f, max_f)
                    axs[i,j].set_title(feature_list[j+3])
                    axs[i,j].set_xlabel("Numbers of weeks")
                    axs[i,j].set_xticks(x_span)
                    axs[i,j].grid()

                    
            axs[0,2].legend(frameon= False, bbox_to_anchor=(1 , 0.2, 0.4, 0.2), loc=3,ncol=1, mode="expand", borderaxespad=0.2)
            
        #axs[2,2].legend( )
        naming = f'dynamic_plot_{name}.png'
        plt.savefig(naming, dpi = 500)  
    
    def best_k_clusters (self, sil_avg_scores_list, dict_max_silhou_score):
    # select the best k where each cluster's peak overshoot the silhoutte_avg cluster
        '''
        This is a function that determines the best number of clusters based on the average silhouette scores data
        based on the number of clusters
        sil_avg_scores_list = average silhouette scores list
        self.clus_num = numbers of clusters
        dict_max_silhou_score = dictionary that stores maximum k_ with maximum scores
        '''
        cluster_range = list(range(2, self.clus_num+1)) 
        
        list(zip(cluster_range, sil_avg_scores_list))
        
        
        i_shift = 1 #shift  
        silhouette_avg_scores = sil_avg_scores_list[i_shift::] #slice the average silhouette scores list
        silh_sort_by_index = np.argsort(silhouette_avg_scores) #recode the list into  index
        silh_sort_by_index_plus_shift = (silh_sort_by_index+cluster_range[0]+ i_shift)[::-1] #sort from behind
    
        data_silh = np.array(sil_avg_scores_list, dtype=float) #convert silhoutte scores to float
        data_silh_diff = np.diff(data_silh) #compute discrete difference of the silhouette score
        data_silh_zero_crossings = np.where(np.diff(np.sign(data_silh_diff)))[0] #eliminating zero from the list
        data_silh_idd_sign = data_silh_zero_crossings[0]+1

        for sil_ind in silh_sort_by_index_plus_shift:
            # print(sil_ind )
            
            # k = best_k
            sil_ind_list = [sil_ind  for n in range(sil_ind)]
            idd = list(zip(sil_ind_list, range(sil_ind )))
            
            # silhouette_avg
            sil_scores = [dict_max_silhou_score.get(key) for key in idd] 

            if any(sil_scores<silhouette_avg_scores[sil_ind-2-i_shift]) or sil_ind<cluster_range[data_silh_idd_sign]:
                if sil_ind==silh_sort_by_index_plus_shift[-1]:
                    best_k = sil_ind
                continue
            else:
                best_k = sil_ind 
                
                break
        print(f' The best k-cluster is {best_k}')
        
        return best_k

    def silhouette_plot(self, data_avg, name, best_k,):  
        #fig = plt.figure(figsize=(10,6),facecolor= None, edgecolor='black', frameon=True)
        cluster_range = list(range(2, self.clus_num+1))
        axs  = plt.axes()
        axs.plot(cluster_range, data_avg, marker = "*", linestyle = "dashed", linewidth = 0.7, c = 'g')
        axs.plot(best_k, data_avg[best_k-1], marker = "o",  linewidth = 0.7, c = 'r')
        axs.step(cluster_range, data_avg, marker = "*", linestyle = "solid", linewidth = 0.7, c = 'b')
        axs.set_xticks(cluster_range)
        plt.rcParams.update({'font.size': 15, 'font.weight':200})
        axs.set_xlabel('Number of clusters',)
        axs.set_ylabel(f'Average scores {name}')
        axs.grid()
        naming = f'silhouette_plot_{name}.png'
        plt.savefig(naming, dpi = 500)  

    def silhouette_score_clus(self, data, name):
        
        '''
        function that estimates silhouette score
        '''
        
        df_fea = self.feature_engineering(data)
        
        r_state = 0 #540 # random seed ?
        init_ = 'k-means++'     # init_ = 'random'
        algorithm_ = 'auto'    # algorithm_ = 'full'
        # algorithm_ = 'elkan'
        verbose_ =  0 #1
        n_init_ = 100 #10
        max_iter_ = 1000 #300
        tol_= 1e-40 #1e-4
        
        inertia_si = []
        silhouette_avg_n_clusters = []
        ith_cluster_silhouette_max_values = {}


        for i in range(2,self.clus_num+1):
            Si_c_kmeans =KMeans(n_clusters= i, init=init_, random_state= r_state,
                            algorithm=algorithm_, verbose=verbose_, n_init=n_init_, 
                            max_iter=max_iter_, tol=tol_)
            Si_c_kmeans.fit(df_fea.values)
            l = Si_c_kmeans.inertia_
            inertia_si.append(l)
            cluster_labels_1 = Si_c_kmeans.fit_predict(df_fea.values)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(df_fea.values, cluster_labels_1)

            silhouette_avg_n_clusters.append(silhouette_avg)
            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(df_fea.values, cluster_labels_1)
            y_lower = 10
            for ii in range(i):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values =\
                    sample_silhouette_values[cluster_labels_1==ii]

                ith_cluster_silhouette_values.sort()
                ith_cluster_silhouette_max_values[i, ii] = max(ith_cluster_silhouette_values)

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples
            # optimal number of clusters

        best_k = self.best_k_clusters(silhouette_avg_n_clusters, ith_cluster_silhouette_max_values)
        # print(f' The length of averages Silhouette is {len(silhouette_avg_n_clusters)}')
        # print(f' The best k-cluster is {best_k}')
        self.silhouette_plot(silhouette_avg_n_clusters, name, best_k)
        return best_k
    
    def optimal_model(self,data, sil_best_k):
        '''
        functions to determine optimal model 
        '''
        best_k = sil_best_k
        r_state = 0 #540 # random seed ?
        init_ = 'k-means++'     # init_ = 'random'
        algorithm_ = 'auto'    # algorithm_ = 'full'
        # algorithm_ = 'elkan'
        verbose_ =  0 #1
        n_init_ = 100 #10
        max_iter_ = 1000 #300
        tol_= 1e-40 #1e-4
        
        X = self.feature_engineering(data)
        X = X.values
        model = KMeans(n_clusters= best_k, init=init_, random_state= r_state,  algorithm=algorithm_, verbose= verbose_, n_init=n_init_, 
                            max_iter=max_iter_, tol=tol_)
        model.fit(X)
        inertia = model.inertia_
        cluster_labels = model.labels_ #s_
        cluster_centroids = model.cluster_centers_
        cluster_centroids = pd.DataFrame(cluster_centroids)

        return inertia , cluster_labels, cluster_centroids

    def Plot_sensor_locs(self, data , Label, ax, m, color, linestyle='none'):
        
        marker = m 
        ax.plot(data.iloc[:, 2], data.iloc[:, 1], marker, label=Label, fillstyle='none', linestyle=linestyle, color = color) #marker = m, label = Label, ms = 2)
        plt.yticks([2.94, 4.34, 3.64, 3.64, 0.89, 0.17, 6.39, 7.11], ['A', 'B', 'C-D', 'C-D', 'E', 'F', 'G', 'H'])
        plt.xticks(np.multiply(3,[0, 1, 2, 3, 4, 5, 6]), ['1', '2', '3', '4', '5', '6', '7'])
    
        return ax

    def polygon_encircle(self, data, ax=None, **kw):
        
        x= data.iloc[:, 2] #x -direction
        y = data.iloc[:, 1] #y -direction 

        y = list(y) # list of y cordinates
        x = list(x) #list of x cordinates

        r_y = y.count(y[0]) == len(y)
        r_x = x.count(x[0]) == len(x)

        # data_locs_len = len(self.data_locs) 
        # print(f'No. of points in cluster: {len(data) }')
        data_locs_len = len(data) 

        if r_y or r_x or data_locs_len<3:
            return ax
        else:
            if not ax: ax = plt.gca()
            p = np.c_[x,y]
            hull = ConvexHull(p, qhull_options='QJ')
            myAbstractPathEffect = [AbstractPathEffect(offset=(0, 0))]
            poly = plt.Polygon(p[hull.vertices,:], fill=True, alpha=0.1, capstyle='round', 
                                joinstyle='miter', linestyle ='--', path_effects =	myAbstractPathEffect, **kw)
            ax.add_patch(poly)
            
     

    def polygon_plot_imp_and_sensor_clusters (self, data, labels, sil_best_k, colornames, markers, name):
        best_k = sil_best_k #best k
        #fig = plt.figure(num=None, figsize=[10,6], dpi=100, facecolor=None, edgecolor=None, frameon=True)
        ax = plt.axes()

        Sensor_clusters = {}

        for m in (range(0, best_k)):
            # indices for cluster m
            idx_m = list(map(list, np.where(labels==m)))[0]
            # print(idx_m)
            X = data.iloc[idx_m,:] # data values of coordinates

            Sensor_clusters[m] = list(X.id) #index of xluster sensors

            label  = f'Cluster {m+1}' 
            color = colornames[m]
            marker = markers[m] #'d'
            ax = self.Plot_sensor_locs(X, label, ax, marker, color) #, linestyle= 'dashed')
            # encircle2(X, ax=ax,  ec=color, fc="none")

            self.polygon_encircle(X, ax=ax,  ec=color, fc=color) 
            ax.legend(frameon=False)
            plt.rcParams.update({'font.size':15, 'font.weight': 200})
            ax.set_xlabel('Sensor locations')
            ax.set_ylabel('Greenhouse sections')

            ax.legend(frameon= False, bbox_to_anchor=(0., 1.02, 1., .102), loc=4,
                    ncol= 4, mode="expand", borderaxespad=0., )
        
        naming = f'polygon_plot_{name}.png'
        plt.savefig(naming, dpi = 300)  

        return Sensor_clusters

    def installed_sensor_locations(self, data, labels, sil_best_k,name):
        
        best_k = sil_best_k
        colors = cm.nipy_spectral(range(56))  # edit this to take in sensor number
        # colors_ = mcolors.CSS4_COLORS
        # colors_ = mcolors.TABLEAU_COLORS
        # colornames = list(colors_)[40::]
        colornames = ['tab:blue', 'tab:orange',  'tab:green',  'tab:red',  'tab:purple',  'tab:brown',
        'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'yellow', 'darkred', 'magenta',
        'indigo', 'yellowgreen', 'tab:olive', 'tab:cyan', 'yellow', 'darkred', 'magenta',
        'indigo', 'yellowgreen','orangered', 'crimson']

        markers = ['o',  'x', '+', 'v', '^', 
                '<', '>', 's', 'd', 'p', 
                'h',  'D', 'H', '8','1', 
                '2', '3', '4', 'P', 'd', 
                'T', 'u' ,'*' ]

        sensors_clusters = self.polygon_plot_imp_and_sensor_clusters(data, labels, best_k, colornames, markers, name)

        return sensors_clusters

    def optimal_sensor_locations(self,data, best_k,clus_sensors, clus_centroids):
        '''
        data means processed data

        '''
        clusters_eds  = dict() # clusters eculidiian distance
        optimal_sensors_id = list() # id of optimal sensor 
        for i in range(best_k):
            ith_clus_sensors_id = clus_sensors[i]
            ith_real_data = data.loc[clus_sensors[i],: ].to_numpy()

            ith_clus_sensors_centroid = clus_centroids.iloc[i].to_numpy().reshape(-1,1).T

            ith_cluster_eds = euclidean_distances(ith_real_data, ith_clus_sensors_centroid)

            clusters_eds[i] = ith_cluster_eds

            ith_cluster_id = int(np.where(ith_cluster_eds == ith_cluster_eds.min())[0])

            optimal_sensors_id.append(ith_clus_sensors_id[ith_cluster_id])

        print(optimal_sensors_id)  
        return    optimal_sensors_id

    def locs_model (self):

        r_state = 0 #540 # random seed ?
        init_ = 'k-means++'     # init_ = 'random'
        algorithm_ = 'auto'    # algorithm_ = 'full'
        # algorithm_ = 'elkan'
        verbose_ =  0 #1
        n_init_ = 100 #10
        max_iter_ = 1000 #300
        tol_= 1e-40 #1e-4
        locs_model = KMeans(n_clusters=1, init=init_, random_state= r_state,
                        algorithm=algorithm_, verbose=0, n_init=n_init_, 
                        max_iter=max_iter_, tol=tol_)
        locs_model.fit(self.data_locs.iloc[:, 1::].to_numpy())
        locs_inertia = locs_model.inertia_
        locs_model_centroid = locs_model.cluster_centers_
        locs_model_centroid_df = pd.DataFrame(locs_model_centroid) 
        return  locs_model_centroid_df

    ######### 
    def predicted_loc_ed (self, opt_clus_centroid,inter_v_grid_df, v_grid_df):
        idxs_pred_ed = []
        pred_locs_ed = {}
        ed_mins = []
        ith = 0
        for ith in range(opt_clus_centroid.shape[0]):
            grid_inter_data= inter_v_grid_df.values #interpolated grid data
            data_centroid_ith = opt_clus_centroid.loc[ith, :].to_numpy().reshape(-1,1).T
            my_eds = euclidean_distances(grid_inter_data,  data_centroid_ith) #eculidian distance

            i_min = np.argsort(my_eds.ravel())[0]  #sort for index 
            idxs_pred_ed.append(i_min)
            ed_min = my_eds[i_min].ravel()
            ed_mins.append(ed_min)
            ed_locs_id = np.where(my_eds==ed_min)[0]

            r_state = 0 #540 # random seed ?
            init_ = 'k-means++'     # init_ = 'random'
            algorithm_ = 'auto'    # algorithm_ = 'full'
            # algorithm_ = 'elkan'
            verbose_ =  0 #1
            n_init_ = 100 #10
            max_iter_ = 1000 #300
            tol_= 1e-40 #1e-4

            locs_model_pred = KMeans(n_clusters=1, init=init_, random_state= r_state,
                                algorithm=algorithm_, verbose=0, n_init=n_init_, 
                                max_iter=max_iter_, tol=tol_)
            locs_model_pred.fit(v_grid_df.loc[ed_locs_id, :].to_numpy())
            loc_inertia = locs_model_pred.inertia_
            loc_model_centroid = locs_model_pred.cluster_centers_
            loc_model_centroid_df = pd.DataFrame(loc_model_centroid)

            pred_locs_ed[ith] = loc_model_centroid.ravel().tolist()

            #print(f'Index [{i_min}] -- Minimum ED: {ed_min}, at coordinate {loc_model_centroid.ravel()}')

        pred_locs_ed = pd.DataFrame.from_dict(pred_locs_ed, orient='index')
        return pred_locs_ed, ed_mins, idxs_pred_ed

    def locations_vol_interp(self, data,dd_locs):

        # create volume grid 
       
        pi = np.pi
        r_xx = round(dd_locs.iloc[:,1].mean(),2)
        th = np.linspace(pi/2, -pi/2, 100);
        Rvec = np.linspace(0, r_xx, 150)
        xyz_use = np.empty(3)
        for R in Rvec:
            # R = 3.64;
            x = R*np.cos(th) ;
            y = R * np.sin(th) + R ;

            number_of_section = self.sections
            Numrows = np.linspace(0, 3*number_of_section, 100)
            for i in Numrows:
                # zuse
                zuse = (i)* np.ones(len(x)) 
                # Array to be added as column
                xyz = np.row_stack((y, zuse, x))
                # Adding column to numpy array
                xyz_use = np.vstack((xyz_use, np.atleast_2d(xyz).T))

        # Data for a three-dimensional Quonsent Lines
        xline = xyz_use[1:, 0].reshape(-1,1).ravel()
        yline = xyz_use[1:, 1].reshape(-1,1).ravel()
        zline = xyz_use[1:, 2].reshape(-1,1).ravel()


        # coordinates for volume grid
        points_xyz = [tuple((x, y, z)) for x,y,z in zip(xline, yline, zline)]
        v_grid_df = pd.DataFrame(points_xyz)

        # interpolate within the volume grid using nearest neighbour interpolator
        arr = self.data_locs.iloc[:, 1::].to_numpy()
        data_set = self.feature_engineering(data)

        interp__1 =  interpolate.NearestNDInterpolator(arr, data_set) 
        interp_grid_df = pd.DataFrame(interp__1(v_grid_df))
        
        #volume grid, interpolated and xyz data
        return v_grid_df, interp_grid_df

    def plot_greenhouse_sensor_locs(self, FigNum):
        # data: locations centroid from kmeans
        Sensor_Locations = self.data_locs
        
        pi = np.pi ;

        th = np.linspace(pi/2, -pi/2, 100);
        R = 3.64;
        x = R*np.cos(th) ;
        y = R * np.sin(th) + R ;

        xyz_use = np.empty(3)

        Numrows = np.arange(7)
        for i in Numrows:
            zuse = (i)* 3* np.ones(len(x))
            # Array to be added as column
            xyz = np.row_stack((y, zuse, x))
            # Adding column to numpy array
            xyz_use = np.vstack((xyz_use, np.atleast_2d(xyz).T))
                # Plot the Sensor Locations
        # fig = plt.figure(num= FigNum , figsize=(20,10), facecolor = None);
        #fig = plt.figure(num=None, figsize=(8,8), dpi=100, facecolor=None, edgecolor=None, frameon=True)
        ax = plt.axes(projection='3d');

        # Data for a three-dimensional Quonsent Lines
        xline = xyz_use[1:, 0]
        yline = xyz_use[1:, 1]
        zline = xyz_use[1:, 2]
        ax.plot3D(xline, yline, zline, '--', label= None);

        # Data for three-dimensional scattered points
        zdata = Sensor_Locations['z']
        xdata = Sensor_Locations['x']
        ydata = Sensor_Locations['y']
        ax.scatter3D(xdata, ydata, zdata, s=60, c = 'blue' , depthshade = True, label ='Installed Sensors');
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Sensor Locations')
        plt.rcParams.update({'font.size': 15, 'font.weight':200})
        #ax.legend('','Working Sensors')
        # ax.legend('','Working Sensors')
        return ax
    
    # Plot the Sensor Location
    def plot_installed_optimal_sensors(self, data_loc_centroid, name):
        #data : location centroid from kmeans
        ax = self.plot_greenhouse_sensor_locs( 1)
        zdata = data_loc_centroid.loc[:, 2]
        xdata = data_loc_centroid.loc[:, 0]
        ydata = data_loc_centroid.loc[:, 1]
        ax.scatter3D(xdata, ydata, zdata, s=100, c = 'red', depthshade = True, label ='Optimal Locations', marker= 's')

        ax.legend(frameon= False, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                ncol=2, mode="expand", borderaxespad=1.5)
        
        naming = f'install_opt_sensor_{name}.png'
        plt.savefig(naming, dpi = 300)  
            
    def polygon_plot_optimal(self, labels, best_k, colornames, markers, sensor_id, pred_ed):

        #fig = plt.figure(num=None, figsize=[10,6], dpi=100, facecolor=None, edgecolor=None, frameon=True)
        ax = plt.axes()

        Sensor_clusters = {}

        for m in (range(0, best_k)):
            # indices for cluster m
            idx_m = list(map(list, np.where(labels==m)))[0]
            # print(idx_m)
            XX = self.data_locs.iloc[idx_m,:]# data values of coordinates

            Sensor_clusters[m] = list(XX.id) #index of xluster sensors

            label  = f'Cluster {m+1}' 
            color = colornames[m]
            marker = markers[m] #'d'
            
            # encircle2(X, ax=ax,  ec=color, fc="none")
            self.polygon_encircle(XX, ax=ax,  ec=color, fc=color) 

            idx_mm = [int(np.where(self.data_locs.id==sensor_id[m])[0])]
            XX_opt = self.data_locs.iloc[idx_mm,:]
            label  = f'Installed {m+1}'
            marker = 'D' #markers[-1] #'d'

            ax = self.Plot_sensor_locs(XX_opt, label, ax, marker, color)
            label = f'Predicted {m+1}'
            XX_star = pred_ed
            # ax = PlotLOCS(XX_star, label, ax, '*', 'red')
            ax.plot(XX_star.iloc[m, 1], XX_star.iloc[m, 0], marker='D', label=label, color = color) 

            ax.legend(frameon=False)
            plt.rcParams.update({'font.size':15, 'font.weight': 200})
            ax.set_xlabel('Sensor locations')
            ax.set_ylabel('Greenhouse sections')

            ax.legend(frameon= False, bbox_to_anchor=(0., 1.02, 1., .102), loc=4,
                    ncol= 4, mode="expand", borderaxespad=0., )
        
        '''naming = f'{self.path[:-3]}_polygon_optimal_{name}.png'
        plt.savefig(naming, dpi = 500)  '''

        return Sensor_clusters

    def plot_optimal_sensor_locations(self, labels, best_k, sensor_id, pred_ed, name):
        '''
        lables => class of sensors from kmeans model
        best_k => optimal number of cluster
        pred_ed => eculidian distance of predicted data
        '''
        colors = cm.nipy_spectral(range(56))
        # colors_ = mcolors.CSS4_COLORS
        # colors_ = mcolors.TABLEAU_COLORS
        # colornames = list(colors_)[40::]
        colornames = ['tab:blue', 'tab:orange',  'tab:green',  'tab:red',  'tab:purple',  'tab:brown',
        'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'yellow', 'darkred', 'magenta',
        'indigo', 'yellowgreen', 'tab:olive', 'tab:cyan', 'yellow', 'darkred', 'magenta',
        'indigo', 'yellowgreen','orangered', 'crimson']

        markers = ['o',  'x', '+', 'v', '^', 
                '<', '>', 's', 'd', 'p', 
                'h',  'D', 'H', '8','1', 
                '2', '3', '4', 'P', 'd', 
                'T', 'u' ,'*' ]

        a = self.polygon_plot_optimal (labels, best_k, colornames, markers, sensor_id, pred_ed)
        
        naming = f'polygon_opt_sensor_{name}.png'
        plt.savefig(naming, dpi = 500)  
        return None
        
    def inst_pred_eculidian(self, data_af, centroids, sensors_id,ed):
        '''
        data_af => transformed real data
        centroids => centroids of optimal sensor from model
        sensors_id => optimal sensor id
        '''
        # compare ecuclidean distance of data of the 
        # predicted locations with the installed locations
        installed_points = self.feature_engineering(data_af.loc[sensors_id,:]).values
        centroids_val = centroids.values
        my_eds_ = euclidean_distances(installed_points, centroids_val)
        Installed_locs_eds = np.diagonal(my_eds_)
        Predicted_locs_eds = np.array(ed)
        #ed_df = pd.concat([pd.DataFrame(Installed_locs_eds ), pd.DataFrame(Predicted_locs_eds, )],axis=1, ignore_index=True, )
        #ed_df = ed_df.rename(columns ={0:'Installed_Ed',1: 'Predicted_Ed'})
        return Installed_locs_eds, Predicted_locs_eds
    
    def index_of_agreement(self, centroids, opt_s_data, opt_id):
        '''
        Centroid => optimal sensors centroid from k-menas
        opt_s_data => data from dummy data for prediction 
            and real data for installation
        opt_id => optimal id for installed and predicted
        '''
        optimal_sensors_points = opt_s_data.loc[opt_id, :]

        Matrix_A = centroids 
        Matrix_B = optimal_sensors_points

        A = Matrix_A.to_numpy().ravel() # convert to vector
        B = Matrix_B.to_numpy().ravel() # convert to vector
        IA = 1-distance.cosine(A,B)     # IA
    
        return IA
    def model_action(self, data, name):

        df_pr = self.data_preprocessing(data) #data preprocessing standardization and normalizer
        df_af = self.applied_feature_extraction(df_pr) #Feature engineering
        self.dynamic_plot(df_af, name)  #dynamic plot
        best_k = self.silhouette_score_clus(df_af, name) #besk
        opt_inertia, opt_labels, opt_centroids = self.optimal_model(df_af, best_k)
        self.path_locs.seek(0)
        data_locs = pd.read_csv(self.path_locs) #reloading data locs
        sen_dict = self.installed_sensor_locations(data_locs, opt_labels, best_k,name) # sensor label and members
        sen_id = self.optimal_sensor_locations(df_af,best_k, sen_dict, opt_centroids)
        #locs_center = self.locs_model()
        v_grid , interp_grid = self.locations_vol_interp(df_af,data_locs)
        pred_ed, locs_ed, pred_id = self.predicted_loc_ed(opt_centroids, interp_grid, v_grid)
        self.plot_installed_optimal_sensors(pred_ed,name)
        self.plot_optimal_sensor_locations(opt_labels, best_k, sen_id, pred_ed,name)
        install_ed, pred_ed = self.inst_pred_eculidian(df_af,opt_centroids,sen_id,locs_ed)
        predicted_IA = self.index_of_agreement(opt_centroids, interp_grid, pred_id)
        installed_IA = self.index_of_agreement(opt_centroids, df_af, sen_id)
        return sen_id, install_ed, pred_ed, installed_IA, predicted_IA
    
    def model (self):
        df = self.data_loading() # data
        df_pt= self.pyschometric_tranformation(df)
        self.path_locs.seek(0)
        data_locs = pd.read_csv(self.path_locs)
        element = list()
        
        Predicted_IAs = list()
        Installed_IAs = list()
        
        
        #ed_df = pd.DataFrame(index= len(self.sensor_ids)) #eculidian dataframe
        required_param_id = []
        ed_df = pd.DataFrame() # initialize ed_df

        for i in self.params:
            if i == 'temp':
                print(f'================================{i.upper()}=======================================')
                df = df_pt[0] #needed data
                
                sen_id, install_ed, pred_ed, installed_IA, predicted_IA = self.model_action(df,i)
                element.append(i)
                Predicted_IAs.append(predicted_IA)
                Installed_IAs.append(installed_IA)
                a1 = pd.DataFrame()
                a1[f'In_{i}'] = install_ed
                a1[f'Pr_{i}'] = pred_ed
                a1[f'{i}_sn'] = sen_id
                required_param_id.append(0)
                ed_df = pd.concat([ed_df, a1])
            elif i == 'humid':
                print(f'================================{i.upper()}=======================================')
                df = df_pt[1] #needed data
                
                sen_id, install_ed, pred_ed, installed_IA, predicted_IA = self.model_action(df,i)
                element.append(i)
                Predicted_IAs.append(predicted_IA)
                Installed_IAs.append(installed_IA)
                a2 = pd.DataFrame()
                a2[f'In_{i}'] = install_ed
                a2[f'Pr_{i}'] = pred_ed
                a2[f'{i}_sn'] = sen_id
                required_param_id.append(1)
                ed_df = pd.concat([ed_df, a2])
            elif i == 'dewpt':
                print(f'================================{i.upper()}=======================================')
                df = df_pt[2] #needed data
                
                sen_id, install_ed, pred_ed, installed_IA, predicted_IA = self.model_action(df,i)
                element.append(i)
                Predicted_IAs.append(predicted_IA)
                Installed_IAs.append(installed_IA)
                a3 = pd.DataFrame()
                a3[f'In_{i}'] = install_ed
                a3[f'Pr_{i}'] = pred_ed
                a3[f'{i}_sn'] = sen_id
                required_param_id.append(2)
                ed_df = pd.concat([ed_df, a3])
            elif i == 'entha':
                print(f'================================{i.upper()}=======================================')
                df = df_pt[3] #needed data
                
                sen_id, install_ed, pred_ed, installed_IA, predicted_IA = self.model_action(df,i)
                element.append(i)
                Predicted_IAs.append(predicted_IA)
                Installed_IAs.append(installed_IA)
                a4 = pd.DataFrame()
                a4[f'In_{i}'] = install_ed
                a4[f'Pr_{i}'] = pred_ed
                a4[f'{i}_sn'] = sen_id
                required_param_id.append(3)
                ed_df = pd.concat([ed_df, a4])
            elif i == 'spev':
                print(f'================================{i.upper()}=======================================')
                df = df_pt[4] #needed data
                
                sen_id, install_ed, pred_ed, installed_IA, predicted_IA = self.model_action(df,i)
                element.append(i)
                Predicted_IAs.append(predicted_IA)
                Installed_IAs.append(installed_IA)

                a5 = pd.DataFrame()
                a5[f'In_{i}'] = install_ed
                a5[f'Pr_{i}'] = pred_ed
                a5[f'{i}_sn'] = sen_id
                required_param_id.append(4)
                ed_df = pd.concat([ed_df, a5])
            elif i == 'humr':
                print(f'================================{i.upper()}=======================================')
                df = df_pt[5] #needed data
                
                sen_id, install_ed, pred_ed, installed_IA, predicted_IA = self.model_action(df,i)
                element.append(i)
                Predicted_IAs.append(predicted_IA)
                Installed_IAs.append(installed_IA)
                a6 = pd.DataFrame()
                a6[f'In_{i}'] = install_ed
                a6[f'Pr_{i}'] = pred_ed
                a6[f'{i}_sn'] = sen_id
                required_param_id.append(5)
                ed_df = pd.concat([ed_df, a6])
            elif i == 'comb':
                print(f'================================{i.upper()}=======================================')
                df = df #needed data
                
                sen_id, install_ed, pred_ed, installed_IA, predicted_IA = self.model_action(df,i)
                element.append(i)
                Predicted_IAs.append(predicted_IA)
                Installed_IAs.append(installed_IA)
                a7 = pd.DataFrame()
                a7[f'In_{i}'] = install_ed
                a7[f'Pr_{i}'] = pred_ed
                a7[f'{i}_sn'] = sen_id
                required_param_id.append(6)
                ed_df = pd.concat([ed_df, a7])
            else:
                pass
        #print(predicted_IA, installed_IA)
        
        # ed_df = pd.concat([a1, a2, a3, a4, a5, a6, a7])  #ed
        ia_df = pd.DataFrame([Predicted_IAs,Installed_IAs, element])#eindex of agreement df
        ed_df.to_csv('ed_df.csv')
        ia_df.to_csv('ia_df.csv')

        
