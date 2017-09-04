import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import csv

img_path='D:\\flower\\example\\test\\' 
model_path='D:\\workspace\\py33\\model-tmp\\'
pred_path='D:\\flower\\model\\cnn\\predictnum.csv'

h=85
w=56
c=1
pic_num = 200
name=['gDiffuse', 'gSpecular', 'bDiffuse', 'bSpecular']
 
def read_img(path):
    imgs=[]
    for i in range(1,pic_num):
        im = path+'%d.bmp'%i
        print('reading the images:%s'%(im))
        img=mpimg.imread(im)
        img=img/255
        img=np.resize(img,(h,w,c))  #因为通道是1所以要设置c的值
        imgs.append(img)
    return np.asarray(imgs,np.float32)
data=read_img(img_path)
    
with tf.Session() as sess:
   
    saver = tf.train.import_meta_graph(model_path+'model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint(model_path))
   
    graph = tf.get_default_graph()
    
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:data}
   
     
    
    logits = graph.get_tensor_by_name("logits_eval:0")
   
    classification_result = sess.run(logits,feed_dict)
   
    #打印出预测矩阵
    
    print(classification_result)
    
    #获取的值写入csv文件
    csvfile_predictnum = open(pred_path,'w',newline='') # 设置newline，否则两行之间会空一行
    writer_predictnum = csv.writer(csvfile_predictnum)
    #第一行写名字
    writer_predictnum.writerow(name) 
    for i in range(len(classification_result)):
        writer_predictnum.writerow(classification_result[i])
    csvfile_predictnum.close()