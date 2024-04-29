import argparse

import random
import json
import torch
import numpy as np
from recbole.quick_start import run_recbole
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt
import  matplotlib
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='COCL', help='name of models')  #Amazon_Sports_and_Outdoors Amazon_Toys_and_Games
    parser.add_argument('--dataset', '-d', type=str, default='Amazon_Beauty', help='name of datasets') #Amazon_Beauty
    parser.add_argument('--config_files', type=str, default='seq.yaml', help='config files')  #ml-1m     Amazon_Clothing_Shoes_and_Jewelry Amazon_Video_Games

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    #run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
    '''model_file ='/root/data1/RecBole-DA-master/log/SASRec/Amazon_Video_Games/bs2048-lmdNone-semNone-None-Jun-17-2023_11-10-38-lr0.002-l20-tauNone-None-DPh0.5-DPa0.5/model.pth'
    #/root/data1/RecBole-DA-master/log/COCL/Amazon_Video_Games/bs2048-lmdNone-semNone-None-Jun-16-2023_07-11-08-lr0.0005-l20-tauNone-None-DPh0.5-DPa0.5/model.pth'
    #'/root/data1/RecBole-DA-master/log/SASRec/Amazon_Video_Games/bs4096-lmdNone-semNone-None-Jun-17-2023_04-21-13-lr0.002-l20-tauNone-None-DPh0.5-DPa0.5/model.pth'
    #/root/data1/RecBole-DA-master/log/DuoRec/Amazon_Video_Games/bs2048-lmd0.1-sem0.1-us_x-Jun-10-2023_06-41-34-lr0.002-l20-tau1-dot-DPh0.5-DPa0.5
    #/root/data1/RecBole-DA-master/log/COCL/Amazon_Video_Games/bs2048-lmdNone-semNone-None-Jun-18-2023_07-56-28-lr0.0005-l20-tauNone-None-DPh0.5-DPa0.5   new
    checkpoint = torch.load(model_file)
    dict = checkpoint['state_dict']
    for param in dict:
        if param =='item_embedding.weight':
            item_embeddings = dict[param].cpu().numpy()
    item_num=np.size(item_embeddings,0)
    #indices2 = list(range(0, item_num,1))
    #sampled_indices2 = random.sample(indices2,2000)
    #sampled_indices2 = torch.tensor(sampled_indices2)
    item_embeddings_sample = torch.tensor(item_embeddings)
    #item_embeddings_sample = torch.index_select(item_embeddings_sample, 0, sampled_indices2)
    label_item = np.ones(item_num,dtype=np.int64)
    #label_test = np.array(label_test1)
    #tsne = TSNE(n_components=2, init='pca', random_state=0)
    tsne = PCA(n_components=2)
    tsne_obj=tsne.fit_transform(item_embeddings_sample)
    tsne_df =pd.DataFrame({'X':tsne_obj[:,0],
                        'Y':tsne_obj[:,1],
                        'digit': 1})
    #fig = plt.figure(figsize=(10, 10))+
    ##sns.scatterplot(x="X", y="Y",
        ##            hue="digit",
         ##           palette=['green'],
         ##           legend='full',
          ##          data=tsne_df)
    sns.kdeplot(x="X", y="Y",
                cbar=True,
                fill=True,
                cmap='Greens',
                thresh=0.05,
                n_levels=12,
                data=tsne_df)
    sns.rugplot(tsne_df['X'], color="g", axis='x', alpha=0.5)
    sns.rugplot(tsne_df['Y'], color="r", axis='y', alpha=0.5)

    plt.xlim(-0.6,0.6)
    plt.ylim(-1.0,1.0)
    plt.show()'''



    #SASRec = [0.0915, 0.0916]  #toys
    #ICLRec_sl = [0.0957, 0.0956]
    #ICLRec_ssl = [0.0963, 0.0958]
    #ICLRec = [0.0972, 0.0962]
    #SASRec = [9.15, 9.17]  #toys
    #ICLRec_sl = [9.57, 9.57]
    #ICLRec_ssl = [9.63, 9.59]
   # ICLRec = [9.72, 9.64]




    #SASRec = [0.0689, 0.076]  # beauty
    #ICLRec_sl = [0.0861,0.083] # 带有sl 0.0429
    #ICLRec_ssl = [0.0868, 0.082]  # 0.0435
    #ICLRec = [0.0889,0.085]
    SASRec = [6.89, 7.6]  # beauty
    ICLRec_sl = [8.61,8.3] # 带有sl 0.0429
    ICLRec_ssl = [8.68, 8.2]  # 0.0435
    ICLRec = [8.89,8.5]



    #SASRec = [0.1152, 0.117]
    #ICLRec_sl = [0.1205, 0.1219]
    #ICLRec_ssl = [0.1211, 0.1216]
    #ICLRec = [0.1224, 0.1225]

    #SASRec = [11.52, 11.7]
   # ICLRec_sl = [12.05, 12.19]
   # ICLRec_ssl = [12.11, 12.16]
   # ICLRec = [12.24, 12.25]

    '''SASRec = [0.2477, 0.2476]
    ICLRec_sl = [0.2529, 0.2522]
    ICLRec_ssl = [0.2533, 0.252]
    ICLRec = [0.2542, 0.2529]

    SASRec = [24.77, 24.76]
    ICLRec_sl = [25.29, 25.22]
    ICLRec_ssl = [25.33, 25.2]
    ICLRec = [25.42, 25.29]'''

    labels = ['HR@10', 'NDCG@10']
    plt.rcParams['axes.labelsize'] = 16  # xy轴label的size
    plt.rcParams['xtick.labelsize'] = 20  # x轴ticks的size
    plt.rcParams['ytick.labelsize'] = 16  # y轴ticks的size
    # plt.rcParams['legend.fontsize'] = 12  # 图例的size

    # 设置柱形的间隔
    width = 0.2  # 柱形的宽度
    x1_list = []
    x2_list = []
    x3_list = []
    x4_list = []
  
    for i in range(len(SASRec)):
        x1_list.append(i)
        x2_list.append(i + width)
        x3_list.append(i + width * 2)
        x4_list.append(i + width * 3)

    # 创建图层
    fig, ax1 = plt.subplots()
    ax1.set_title("Beauty", fontsize=16)
    # 设置左侧Y轴对应的figure
    ax1.set_ylabel('HR@10(%)')
    ax1.set_ylim(6.0, 9.5)
    #ax1.set_yticks(np.arange(0.114, 0.047, 0.01))
    p1=ax1.bar(x1_list, SASRec, width=width, color='lightgreen', align='edge')
    p2=ax1.bar(x2_list, ICLRec_sl, width=width, color='deepskyblue', align='edge')
    p3=ax1.bar(x3_list, ICLRec_ssl, width=width, color='orange', align='edge', tick_label = labels)
    p4=ax1.bar(x4_list, ICLRec, width=width, color='limegreen', align='edge')
    ax1.set_xticklabels(ax1.get_xticklabels())  # 设置共用的x轴

    # 设置右侧Y轴对应的figure
    ax2 = ax1.twinx()
    ax2.set_ylabel('NDCG@10(%)')
    ax2.set_ylim(3.4, 4.8)


    plt.tight_layout()
    plt.show()

