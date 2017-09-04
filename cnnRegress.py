import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import time
import os
import csv

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

pic_path='E:\\EMSim\\EMSimulation_fin\\traindata\\T72\\3383\\'
model_path='E:\\EMSim\\yyl\\model\\loss2l2\\'+'model.ckpt'
label_path='E:\\EMSim\\yyl\\entries.csv'
log_path='E:\\EMSim\\yyl\\logs\\loss2l2\\'

#将所有的图片resize成w*h*c
h=85
w=56
c=1
min_loss=10
pic_num = 2705

#读取图片
def read_img(path):
    imgs=[]
    for i in range(1,pic_num):
        im = path+'%d.bmp'%i
#         print('reading the images:%s'%(im))
        img=mpimg.imread(im)
        img=img/255.0;
        img=np.resize(img,(h,w,c))  #因为通道是1所以要设置c的值,因为x是四维的所以要改成4为的形式
        imgs.append(img)
    return np.asarray(imgs,np.float32)
data=read_img(pic_path)

#读取csv返回数据名字和数据
def read_csv(path):
    data = []
    csvfile = open(path, 'r')
    reader = csv.reader(csvfile)
    for it in reader:
        data.append(it)
    return data[:1],np.asarray(data[1:],np.float32)#将名字和数据分别返回
label_name,label = read_csv(label_path)

#打乱顺序
num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]

#将所有数据分为训练集和测试集
ratio=1
s=np.int(num_example*ratio)

x_train=data[:s]
y_train=label[:s]
x_val=data[s:]
y_val=label[s:]

#-----------------构建网络----------------------
#占位符
x=tf.placeholder(tf.float32,shape=[None,h,w,c],name='x')
y_=tf.placeholder(tf.float32,shape=[None,4],name='y_')
# rat=tf.constant([[1.2,0.4,1.2,1.2]], tf.float32, shape=[1,4],name='ratio')
# lr = tf.placeholder(tf.float32, shape=[])

#添加原图到tensorboard images
image_input = tf.reshape(x, [-1, h, w, 1])
tf.summary.image('input', image_input, 2)

#第一个卷积层（85*56——>42*28)
conv1=tf.layers.conv2d(
      inputs=x,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
#添加图像到tensorboard images
p1 = tf.reshape(pool1, [-1, 42 * 28 * 32])
p1_image = tf.reshape(p1, [-1,42, 28, 1])
tf.summary.image("p1_image", p1_image, 2)  #每层显示2个图片

#第二个卷积层(42*28->21*14)
conv2=tf.layers.conv2d(
      inputs=pool1,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
#添加图像到tensorboard images
p2 = tf.reshape(pool2, [-1, 21 * 14 * 32])
p2_image = tf.reshape(p2, [-1,21, 14, 1])
tf.summary.image("p2_image", p2_image, 2)  #每层显示2个图片

#第三个卷积层(21*14->10*7)
conv3=tf.layers.conv2d(
      inputs=pool2,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
pool3=tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
#添加图像到tensorboard images
p3 = tf.reshape(pool3, [-1, 10 * 7 * 64])
p3_image = tf.reshape(p3, [-1,10, 7, 1])
tf.summary.image("p3_image", p3_image, 2)  #每层显示2个图片

#第四个卷积层(10*7->5*3)
conv4=tf.layers.conv2d(
      inputs=pool3,
      filters=128,
      kernel_size=[2, 2],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
pool4=tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
#添加图像到tensorboard images
p4 = tf.reshape(pool4, [-1, 5 * 3 * 128])
p4_image = tf.reshape(p4, [-1,5, 3, 1])
tf.summary.image("p4_image", p4_image, 2)  #每层显示2个图片

re1 = tf.reshape(pool4, [-1, 5 * 3 * 128])


#全连接层
dense1 = tf.layers.dense(inputs=re1, 
                      units=256, 
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
                      
                      
logits= tf.layers.dense(inputs=dense1, 
                        units=4, 
                        activation=tf.nn.sigmoid,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

tf.summary.histogram('logits', logits)
#---------------------------网络结束---------------------------

#(小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
b = tf.constant(value=1,dtype=tf.float32)
logits_eval = tf.multiply(logits,b,name='logits_eval') 

# print(len(tf.trainable_variables())) #返回的是需要训练的变量列表,所有的w，b
dense_w=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) #获取w
regularization_loss = tf.reduce_mean(tf.square(dense_w))
loss1 = tf.reduce_mean(tf.div(tf.abs(y_ - logits),y_))
loss2 = tf.reduce_mean(tf.square(tf.div((y_ - logits),y_)))
loss3 = tf.reduce_mean(tf.square(y_ - logits))
loss4 = loss1 + 0.001*regularization_loss#0.01是正则化系数
# loss = tf.reduce_mean(tf.square(tf.div((y_ - logits),y_))) + 0.01*regularization_loss#0.01是正则化系数
# loss = tf.reduce_mean(tf.square(tf.div((y_ - logits),y_)))
# loss = tf.reduce_mean(tf.square((y_ - logits)*rat))

loss = loss2 + loss4
tf.summary.scalar('loss', loss) 

train_op=tf.train.AdagradOptimizer(learning_rate=0.01).minimize(loss)
#train_op=tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
# train_op=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

acc = tf.div(logits-y_,y_)

tf.summary.scalar('acc1', acc[0][0]) 
tf.summary.scalar('acc2', acc[0][1]) 
tf.summary.scalar('acc3', acc[0][2]) 
tf.summary.scalar('acc4', acc[0][3]) 
#参数的值是保留4位小数的

#定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


#训练和测试数据，可将n_epoch设置更大一些

n_epoch=3000
batch_size=32
saver=tf.train.Saver()
# saver=tf.train.Saver(max_to_keep=1)
sess=tf.Session()
# sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer()) 
writer = tf.summary.FileWriter(log_path, sess.graph)     # write to file
merge_op = tf.summary.merge_all()           
start_time = time.time()
for epoch in range(n_epoch):
    
    #training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _,err,ac,result=sess.run([train_op,loss,acc,merge_op], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err; train_acc += ac; n_batch += 1
        
        if epoch % 10 == 0:
            writer.add_summary(result, epoch)
        
    if epoch % 10 == 0:
        print("第%d个epoch"%epoch)
        
        print("   train loss: %f" % (train_loss/ n_batch))
        print("   train acc1: %f" % (train_acc[0][0]/ n_batch))
        print("   train acc2: %f" % (train_acc[0][1]/ n_batch))
        print("   train acc3: %f" % (train_acc[0][2]/ n_batch))
        print("   train acc4: %f" % (train_acc[0][3]/ n_batch))
    
    if(epoch<2):
          min_loss = train_loss
   
    if(train_loss<min_loss): 
          saver.save(sess, model_path) #保存模型到model-tmp/model.ckpt，注意一定要有一层文件夹，否则保存不成功！！！
          min_loss = train_loss
          
end_time = time.time()
print("time: min")
print((end_time-start_time)/60)
sess.close()