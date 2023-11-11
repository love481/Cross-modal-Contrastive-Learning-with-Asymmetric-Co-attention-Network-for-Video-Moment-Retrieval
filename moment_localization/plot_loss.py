import matplotlib.pyplot as plt
import numpy as np
import csv
rank_1_iou_3=[]
rank_1_iou_5=[]
rank_1_iou_7=[]
rank_5_iou_3=[]
rank_5_iou_5=[]
rank_5_iou_7=[]
rank_1_iou_3_s=[]
rank_1_iou_5_s=[]
rank_1_iou_7_s=[]
rank_5_iou_3_s=[]
rank_5_iou_5_s=[]
rank_5_iou_7_s=[]
iteration=[]
with open('plot_folder/msat_128.txt', 'r') as datafile:
    plotting = csv.reader(datafile)
    
    for ROW in plotting:
        rank_1_iou_3.append(float(ROW[0]))
        rank_1_iou_5.append(float(ROW[1]))
        rank_1_iou_7.append(float(ROW[2]))
        rank_5_iou_3.append(float(ROW[3]))
        rank_5_iou_5.append(float(ROW[4]))
        rank_5_iou_7.append(float(ROW[5]))

with open('plot_folder/skip_128.txt', 'r') as datafile:
    plotting = csv.reader(datafile)
    
    for ROW in plotting:
        rank_1_iou_3_s.append(float(ROW[0]))
        rank_1_iou_5_s.append(float(ROW[1]))
        rank_1_iou_7_s.append(float(ROW[2]))
        rank_5_iou_3_s.append(float(ROW[3]))
        rank_5_iou_5_s.append(float(ROW[4]))
        rank_5_iou_7_s.append(float(ROW[5]))

## for activitynet
# iteration = list(range(5345,21710,333))
# fig, axis = plt.subplots(2,2,figsize=(15, 10))
# axis[0, 0].plot(iteration, rank_1_iou_5, 'tab:blue')
# axis[0, 0].plot(iteration, rank_1_iou_5_s, 'tab:orange')
# axis[0, 0].set(xlabel='iteration', ylabel='Rank@1,mIoU@0.5')
# axis[0,0].legend(['MSAT+Skip_connected','Skip_connected'])
# axis[0, 1].plot(iteration, rank_1_iou_7, 'tab:blue')
# axis[0, 1].plot(iteration, rank_1_iou_7_s, 'tab:orange')
# axis[0, 1].set(xlabel='iteration', ylabel='Rank@1,mIoU@0.7')
# axis[0,1].legend(['MSAT+Skip_connected','Skip_connected'])
# axis[1, 0].plot(iteration, rank_5_iou_5, 'tab:blue')
# axis[1, 0].plot(iteration, rank_5_iou_5_s, 'tab:orange')
# axis[1, 0].set(xlabel='iteration', ylabel='Rank@5,mIoU@0.5')
# axis[1,0].legend(['MSAT+Skip_connected','Skip_connected'])
# axis[1, 1].plot(iteration, rank_5_iou_7, 'tab:blue')
# axis[1, 1].plot(iteration, rank_5_iou_7_s, 'tab:orange')
# axis[1, 1].set(xlabel='iteration', ylabel='Rank@5,mIoU@0.7')
# axis[1,1].legend(['MSAT+Skip_connected','Skip_connected'])
# fig.savefig('Rank@n,mIoU@m_activityNet.png')

##for tacos
iteration = list(range(1398,41940+1398,1398))
fig, axis = plt.subplots(2,2,figsize=(13, 8))
axis[0, 0].plot(iteration, rank_1_iou_3, 'tab:blue')
axis[0, 0].plot(iteration, rank_1_iou_3_s, 'tab:orange')
axis[0, 0].set(xlabel='iteration', ylabel='Rank@1,mIoU@0.3')
axis[0,0].legend(['MSAT+Skip_connected','Skip_connected'])
axis[0, 1].plot(iteration, rank_1_iou_5, 'tab:blue')
axis[0, 1].plot(iteration, rank_1_iou_5_s, 'tab:orange')
axis[0, 1].set(xlabel='iteration', ylabel='Rank@1,mIoU@0.5')
axis[0,1].legend(['MSAT+Skip_connected','Skip_connected'])
axis[1, 0].plot(iteration, rank_5_iou_3, 'tab:blue')
axis[1, 0].plot(iteration, rank_5_iou_3_s, 'tab:orange')
axis[1, 0].set(xlabel='iteration', ylabel='Rank@5,mIoU@0.3')
axis[1,0].legend(['MSAT+Skip_connected','Skip_connected'])
axis[1, 1].plot(iteration, rank_5_iou_5, 'tab:blue')
axis[1, 1].plot(iteration, rank_5_iou_5_s, 'tab:orange')
axis[1, 1].set(xlabel='iteration', ylabel='Rank@5,mIoU@0.5')
axis[1,1].legend(['MSAT+Skip_connected','Skip_connected'])
fig.savefig('Rank@n,mIoU@m_tacos.png')