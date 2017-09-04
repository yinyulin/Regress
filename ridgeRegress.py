import numpy as np
import matplotlib.image as mpimg
import csv
from sklearn import linear_model


# path='D:\\flower\\wnum\\'
path='D:\\flower\\ridgeRegress\\T72\\3340\\simulation\\'
entries_path="D:\\flower\\ridgeRegress\\entries.csv"
out_path='D:\\flower\\ridgeRegress\\T72\\3340\\'
#将所有的图片resize成100*100
h=85
w=56
c=1
pic_num = 2504
#读取图片
def read_img(path):
    imgs=[]
    for i in range(pic_num):
        i = i + 1
        im = path+'%d.bmp'%i
        print('reading the images:%s'%(im))
        img=mpimg.imread(im)
        img=np.resize(img,w*h*c) #把data变成list[[]]每张图片一行，每行的列数是h*w*c
        img=img/255    #归一化
        imgs.append(img)
    return np.asarray(imgs,np.float32)
data=read_img(path)

#读取csv返回数据名字和数据
def read_csv(path):
    data = []
    csvfile = open(path, 'r')
    reader = csv.reader(csvfile)
    for it in reader:
        data.append(it)
    return data[:1],np.asarray(data[1:],np.float32)#将名字和数据分别返回

label_name,label = read_csv(entries_path)

#打乱2504个值的顺序
num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]

#选择回归方式
regr = linear_model.LinearRegression()
# regr=linear_model.LassoCV(alphas=[0.1, 0.5, 1]) #losscv不支持多个输出
# alphas=np.arange(0.01,100,10) 
# regr=linear_model.RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100])
regr.fit(data_train, label_train)
#写出系数
csvfile_coe = open(out_path+'coefficients.csv','w',newline='') # 设置newline，否则两行之间会空一行
writer_coe = csv.writer(csvfile_coe)
writer_coe.writerow(label_name[0])
len_coef = len(regr.coef_)
len_coef_0 = len(regr.coef_[0])
for i in range(len_coef_0):
    tmp = []
    for j in range(len_coef):
        tmp.append(regr.coef_[j][i])
    writer_coe.writerow(tmp)
csvfile_coe.close()
#写出截距
csvfile_intercept = open(out_path+'intercept.csv', 'w', newline='')
writer_intercept = csv.writer(csvfile_intercept)
writer_intercept.writerow(label_name[0])
writer_intercept.writerow(regr.intercept_)
csvfile_intercept.close()


xPred = data_test
yPred = regr.predict(xPred)
print('score')
print(regr.score(xPred,label_test,sample_weight=None))
print ("predicted y: ")

csvfile_testnum = open(out_path+'testnum.csv','w',newline='') # 设置newline，否则两行之间会空一行
writer_testnum = csv.writer(csvfile_testnum)
csvfile_realnum = open(out_path+'realnum.csv','w',newline='') # 设置newline，否则两行之间会空一行
writer_realnum = csv.writer(csvfile_realnum)
len_yPred = len(yPred)
for i in range(len_yPred):
    print ('预测值：',yPred[i])
    print('真实值：',label_test[i])
    print('****************')
    writer_testnum.writerow(yPred[i])
    writer_realnum.writerow(label_test[i])
csvfile_testnum.close()
csvfile_realnum.close()