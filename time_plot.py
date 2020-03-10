
import numpy as np
import matplotlib.pyplot as plt
import pickle

# time
if 1:
    
    # diff k updated 
    # x = [2,3,4,5]
    # y1 = np.array([103.9520228, 108.4128144, 132.9576972, 356.7644398])/60
    # y2 = np.array([130.1090788, 141.955754 ,178.2252692, 398.8512982])/60
    # y3 = np.array([183.2030338, 202.6031953, 275.6688485, 752.6530013])/60

    # diff C
    x=[1,2,3,4]
    y1 = np.array([103.9520228, 150.1303577, 201.1459043, 302.4522453])/60
    y2 = np.array([178.2252692, 235.5056164, 369.8320406, 425.7817956])/60
    y3 = np.array([202.6031953, 273.287533, 350.6019204, 428.1202946])/60

    

    # bar_width = 1
    plt.figure() #创建绘图对象 plt.figure(figsize=(6,4))
    plt.plot(x,y1,"b--",marker='o',markersize=10, linewidth=3, label='IsoNN(HIV-fMRI)')   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x,y2,"r--",marker='s',markersize=10, linewidth=3, label='IsoNN(HIV-DTI)')
    plt.plot(x,y3,"c--",marker='*',markersize=10, linewidth=3, label='IsoNN(BP-fMRI)')
    plt.ylabel("Time(min)", fontsize=16) #X轴标签
    plt.xlabel("c", fontsize=16)  #Y轴标签
    plt.xticks(x, x)
    # plt.xticks(x, ('1', '2', '3', '4'))
    plt.legend(prop={'size': 14}) 
    # plt.title("Line p") #图标题
    plt.show()  #显示图
if 0:
    # fast cmp
    fast = np.array([1744.516926, 3056.97354, 3699.90773, 5085.399735, 6041.823228])/60
    normal = np.array([281.9979355, 471.0994611, 869.1605818, 4827.744245, 33237.35804])/60
    x = [2,3,4, 5, 6]
    plt.figure() #创建绘图对象 plt.figure(figsize=(6,4))
    plt.plot(x,fast,"b--",marker='o',markersize=10, linewidth=3, label='IsoNN-fast')   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x,normal,"c--",marker='s',markersize=10, linewidth=3, label='IsoNN')
    plt.ylabel("Time(min)", fontsize=16) #X轴标签
    plt.xlabel("k", fontsize=16)  #Y轴标签
    plt.xticks(x, x)
    # plt.xticks(x, ('2', '3', '4', '5'))
    plt.legend(prop={'size': 14}) 
    # plt.title("Line p") #图标题
    plt.show()  #显示图
