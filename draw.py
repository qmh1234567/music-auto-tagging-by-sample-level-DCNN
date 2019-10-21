""" 
@ author: Qmh
@ file_name: draw.py
@ time: 2019:10:18:19:01
""" 
import numpy as np
import constants as c
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus']=False #解决负数坐标显示问题

auc_list_D = np.load('./npys/rocuac-DCNN.npy')
auc_list_S = np.load('./npys/rocuac-SampleCNN.npy')
auc_list_DSE = np.load('./npys/rocuac-DCNNse.npy')



def compare_values(list_A,list_B):
    greater_tags = {}
    lower_tags = {}
    for index,auc in enumerate(list_A):
        if auc < list_B[index]:
            greater_tags[c.TAGS[index]] =list_B[index] - auc
        elif auc > list_B[index]:
            lower_tags[c.TAGS[index]] = auc-list_B[index]

    sorted_greater_tags = sorted(greater_tags.items(),key=lambda x:x[1],reverse=True)
    sorted_lower_tags = sorted(lower_tags.items(),key=lambda x:x[1],reverse=True)
    return sorted_greater_tags,sorted_lower_tags



# 画图
def draw_auc_scores(auc_list_D,auc_list_S,auc_list_DSE):
    line_D = np.mean(auc_list_D)
    line_S = np.mean(auc_list_S)
    line_DSE = np.mean(auc_list_DSE)

    mean_value = 0.8

    auc_list_D -= mean_value
    auc_list_S -= mean_value
    auc_list_DSE -= mean_value

    plt.figure(figsize=(80, 25))  # width,height
    bar_width = 0.25
    index = np.arange(len(c.TAGS))
    

    rects_S = plt.bar(index,auc_list_S,bar_width,label='SampleCNN with SE blcoks',color='y',bottom=mean_value)

    rects_D = plt.bar(index+bar_width,auc_list_D,bar_width,label='proposed DCNN',color='r',bottom=mean_value)

    rects_DSE = plt.bar(index+bar_width*2,auc_list_DSE,bar_width,label='proposed DCNN with SE blocks',color='b',bottom=mean_value)

    plt.axhline(y=line_S,ls="-.",c="y",lw=6)#添加水平直线
    plt.axhline(y=line_D,ls="-.",c="r",lw=6)
    plt.axhline(y=line_DSE,ls="-.",c="b",lw=6)

    # plt.ylim(ymax=2,ymin=-2)

    fontsize = 45

    plt.xticks(index+bar_width*2.5,c.TAGS,rotation=30, fontsize=fontsize,ha='right')
    plt.yticks(np.arange(0.65,1,0.1),fontsize=fontsize)

    plt.ylabel("AUC score",fontsize=fontsize)
    plt.legend(fontsize=fontsize)

    plt.savefig("./npys/1.png")

    # for rect in rects:
    #     height = rect.get_height()
    #     plt.text(rect.get_x()+rect.get_width()/2,height,str(height)+'%', ha='center', va='bottom')
    # plt.show()    


if __name__ == "__main__":
    greater_tags,lower_tags = compare_values(auc_list_S,auc_list_D)
    print("greater_tags=",greater_tags)
    print("lower_tags=",lower_tags)
    print(len(greater_tags))
    print(len(lower_tags))
    print("*"*10)
    grater_tags,lower_tags = compare_values(auc_list_S,auc_list_DSE)
    print("greater_tags=",greater_tags)
    print("lower_tags=",lower_tags)
    draw_auc_scores(auc_list_D,auc_list_S,auc_list_DSE)
    print(len(grater_tags))
    print(len(lower_tags))


