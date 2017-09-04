import csv
import numpy as np
import matplotlib.image as mpimg

h=85
w=56
c=1
pic_num = 200
predict_num = []
path_pic = 'D:\\flower\\ridgeRegress\\T72\\3383\\test\\'
path_coefficients = 'D:\\flower\\ridgeRegress\\coefficients.csv'
path_intercept = 'D:\\flower\\ridgeRegress\\intercept.csv'
path_predtion='D:\\flower\\ridgeRegress\\T72\\3383\\predictnum.csv'


#读取图片
def read_img(path):
    imgs=[]
    for i in range(pic_num):
        i = i + 1
        im = path+'%d.bmp'%i
#         print('reading the images:%s'%(im))
        img=mpimg.imread(im)
        img=np.resize(img,(h,w,c))  #因为通道是1所以要设置c的值
        img=np.resize(img,w*h)   #把data变成list[[]]每张图片一行，每行的列数是h*w*c
        img=img / 255.0;
        imgs.append(img)
    return np.asarray(imgs,np.float32)
data=read_img(path_pic)
# data = np.resize(data,[pic_num,h*w*c]) #把data变成list[[]]每张图片一行，每行的列数是h*w*c

#读取csv返回数据名字和数据
def read_csv(path):
    data = []
    csvfile = open(path, 'r')
    reader = csv.reader(csvfile)
    for it in reader:
        data.append(it)
    return data[:1],np.asarray(data[1:],np.float32)#将名字和数据分别返回

coefficients_name,coefficients_num = read_csv(path_coefficients)
intercept_name,intercept_num = read_csv(path_intercept)

#将读取的图片和系数相乘+截距，输出预测值,注意coefficients的数据行列交换
print(intercept_name[0])
for num in range(len(data)):
    predict = []
    for i in range(4):
        tmp = 0
        for j in range(len(coefficients_num)):
            tmp += coefficients_num[j][i]*data[num][j]
        tmp += intercept_num[0][i]
        predict.append(tmp)
    predict_num.append(predict)
#     print('第%i张图'%(1+num))
#     print(predict)
    
#获取的值写入csv文件
csvfile_predictnum = open(path_predtion,'w',newline='') # 设置newline，否则两行之间会空一行
writer_predictnum = csv.writer(csvfile_predictnum)
#第一行写名字
writer_predictnum.writerow(coefficients_name[0]) 
for i in range(len(predict_num)):
    writer_predictnum.writerow(predict_num[i])
csvfile_predictnum.close()