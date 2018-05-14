本篇博客tensorflow1.7,整个项目源码：[github](https://github.com/shawshany/Traffic_sign_Classify)
# 引言
本次博客将分享Udacity无人驾驶纳米学位的另一个项目，交通标志的识别。
本次项目实现主要采用CNN卷积神经网络，具体的网络结构参考Lecun提出的LeNet结构。参考文献：[Lecun Paper](https://download.csdn.net/download/u010665216/10412418)
# 项目流程图
本项目的实现流程如下所示：
![这里写图片描述](https://img-blog.csdn.net/20180514102759447?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

# 代码实现及解释
接下来我们就按照项目流程图来逐块实现,本项目数据集：[German data](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip) 
如果打不开，则有备用链接：[备用](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
```python
#import important packages/libraries
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import random
import csv
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
from skimage import transform as transf
from sklearn.model_selection import train_test_split
import cv2
from prettytable import PrettyTable
%matplotlib inline
SEED = 2018
```

    /home/ora/anaconda3/envs/tensorflow/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters


    WARNING:tensorflow:From /home/ora/anaconda3/envs/tensorflow/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/base.py:198: retry (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use the retry module or similar alternatives.



```python
# 导入数据并可视化
training_file = 'data/train.p'
testing_file = 'data/test.p'

with open(training_file,mode='rb') as f:
    train = pickle.load(f)
with open(testing_file,mode='rb') as f:
    test = pickle.load(f)

X_train,y_train = train['features'],train['labels']
X_test,y_test = test['features'],test['labels']
```

# Dataset Summary and Expoloration
下面我们对德国交通指示牌数据集进行可视化处理


```python
n_train = len(X_train)
n_test = len(X_test)

_,IMG_HEIGHT,IMG_WIDTH,IMG_DEPTH = X_train.shape
image_shape = (IMG_HEIGHT,IMG_WIDTH,IMG_DEPTH)

with open('data/signnames.csv','r') as sign_name:
    reader = csv.reader(sign_name)
    sign_names = list(reader)

sign_names = sign_names[1::]
NUM_CLASSES = len(sign_names)
print('Total number of classes:{}'.format(NUM_CLASSES))

n_classes = len(np.unique(y_train))
assert (NUM_CLASSES== n_classes) ,'1 or more class(es) not represented in training set'

n_test = len(y_test)

print('Number of training examples =',n_train)
print('Number of testing examples =',n_test)
print('Image data shape=',image_shape)
print('Number of classes =',n_classes)
```

    Total number of classes:43
    Number of training examples = 34799
    Number of testing examples = 12630
    Image data shape= (32, 32, 3)
    Number of classes = 43



```python
#data visualization,show 20 images
def visualize_random_images(list_imgs,X_dataset,y_dataset):
    #list_imgs:20 index
    _,ax = plt.subplots(len(list_imgs)//5,5,figsize=(20,10))
    row,col = 0,0
    for idx in list_imgs:
        img = X_dataset[idx]
        ax[row,col].imshow(img)
        ax[row,col].annotate(int(y_dataset[idx]),xy=(2,5),color='red',fontsize='20')
        ax[row,col].axis('off')
        col+=1
        if col==5:
            row,col = row+1,0
    plt.show()
ls = [random.randint(0,len(y_train)) for i in range(20)]
visualize_random_images(ls,X_train,y_train)
```


![png](https://img-blog.csdn.net/20180514102858287?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



```python
def get_count_imgs_per_class(y, verbose=False):
    num_classes = len(np.unique(y))
    count_imgs_per_class = np.zeros( num_classes )

    for this_class in range( num_classes ):
        if verbose: 
            print('class {} | count {}'.format(this_class, np.sum( y  == this_class )) )
        count_imgs_per_class[this_class] = np.sum(y == this_class )
    #sanity check
    return count_imgs_per_class
class_freq = get_count_imgs_per_class(y_train)
print('------- ')
print('Highest count: {} (class {})'.format(np.max(class_freq), np.argmax(class_freq)))
print('Lowest count: {} (class {})'.format(np.min(class_freq), np.argmin(class_freq)))
print('------- ')
plt.bar(np.arange(NUM_CLASSES), class_freq , align='center')
plt.xlabel('class')
plt.ylabel('Frequency')
plt.xlim([-1, 43])
plt.title("class frequency in Training set")
plt.show()
sign_name_table = PrettyTable()
sign_name_table.field_names = ['class value', 'Name of Traffic sign']
for i in range(len(sign_names)):
    sign_name_table.add_row([sign_names[i][0], sign_names[i][1]] )
    
print(sign_name_table)
```

    ------- 
    Highest count: 2010.0 (class 2)
    Lowest count: 180.0 (class 0)
    ------- 



![png](https://img-blog.csdn.net/20180514102924375?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


    +-------------+----------------------------------------------------+
    | class value |                Name of Traffic sign                |
    +-------------+----------------------------------------------------+
    |      0      |                Speed limit (20km/h)                |
    |      1      |                Speed limit (30km/h)                |
    |      2      |                Speed limit (50km/h)                |
    |      3      |                Speed limit (60km/h)                |
    |      4      |                Speed limit (70km/h)                |
    |      5      |                Speed limit (80km/h)                |
    |      6      |            End of speed limit (80km/h)             |
    |      7      |               Speed limit (100km/h)                |
    |      8      |               Speed limit (120km/h)                |
    |      9      |                     No passing                     |
    |      10     |    No passing for vechiles over 3.5 metric tons    |
    |      11     |       Right-of-way at the next intersection        |
    |      12     |                   Priority road                    |
    |      13     |                       Yield                        |
    |      14     |                        Stop                        |
    |      15     |                    No vechiles                     |
    |      16     |      Vechiles over 3.5 metric tons prohibited      |
    |      17     |                      No entry                      |
    |      18     |                  General caution                   |
    |      19     |            Dangerous curve to the left             |
    |      20     |            Dangerous curve to the right            |
    |      21     |                    Double curve                    |
    |      22     |                     Bumpy road                     |
    |      23     |                   Slippery road                    |
    |      24     |             Road narrows on the right              |
    |      25     |                     Road work                      |
    |      26     |                  Traffic signals                   |
    |      27     |                    Pedestrians                     |
    |      28     |                 Children crossing                  |
    |      29     |                 Bicycles crossing                  |
    |      30     |                 Beware of ice/snow                 |
    |      31     |               Wild animals crossing                |
    |      32     |        End of all speed and passing limits         |
    |      33     |                  Turn right ahead                  |
    |      34     |                  Turn left ahead                   |
    |      35     |                     Ahead only                     |
    |      36     |                Go straight or right                |
    |      37     |                Go straight or left                 |
    |      38     |                     Keep right                     |
    |      39     |                     Keep left                      |
    |      40     |                Roundabout mandatory                |
    |      41     |                 End of no passing                  |
    |      42     | End of no passing by vechiles over 3.5 metric tons |
    +-------------+----------------------------------------------------+



```python
def histograms_randImgs(label,channel,n_imgs=5,ylim=50):
    '''
    Histogram (pixel intensity distribution) for a selection of images with the same label.
    For better visualization, the images are shown in grayscale
    label - the label of the images
    n_imgs - number of images to show (default=5)
    channel - channel used to compute histogram
    ylim - range of y axis values for histogram plot (default=50)
    '''
    assert channel < 3,'image are RGB,choose channel value between in the range[0,2]'
    assert (np.sum(y_train==label))>=n_imgs,'reduce your number of images'
    
    all_imgs = np.ravel(np.argwhere(y_train==label))
    
    #随机选择5张图片
    ls_idx = np.random.choice(all_imgs,size=n_imgs,replace=False)
    _,ax = plt.subplots(n_imgs,2,figsize=(10,10))
    print('Histogram of selected images from the class{} ......'.format(label))
    row,col = 0,0
    for idx in ls_idx:
        img = X_train[idx,:,:,channel]
        #print(img.shape)
        ax[row,col].imshow(img,cmap='gray')
        ax[row,col].axis('off')
        
        hist = np.histogram(img,bins=256)
        ax[row,col+1].hist(hist,bins=256)
        ax[row,col+1].set_xlim([0,100])
        ax[row,col+1].set_ylim([0,ylim])
        col,row = 0,row+1
    plt.show()
histograms_randImgs(38,1)
```

    Histogram of selected images from the class38 ......



![png](https://img-blog.csdn.net/20180514102938518?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


# 接下来对数据做进一步处理
我们完成以下几个步骤：
>* 数据增强
>* 将RGB转换成Grayscale
>* 数据尺度变换

**Note**：数据集的划分必须在数据增强完成前(防止验证集被合成图像污染)
## 数据增强具体步骤
这里的数据增强主要是：1.增加训练集的大小 2.调整了类别分布（类别分布是不均衡的，因为测试集可能相较与训练集来讲，有着不同的分布，因此我们希望在类别分布均衡的数据集上训练，给不同类别相同的权重，然后在不均衡的数据集上测试时可以有更好的效果）
数据增强后，我们得到每个类别4000张图片
数据增强的方法主要就是从原始数据集中随机选取图片，并应用仿射变换
>* 旋转角度我限制在【-10，10】度之间，如果旋转角度过大，有些交通标志的意思可能就会发生变化了
>* 水平、垂直移动的话，范围限制在【-3，3】px之间
>* 伸缩变换限制在【0.8，1.2】


```python
def random_transform(img,angle_range=[-10,10],
                    scale_range=[0.8,1.2],
                    translation_range=[-3,3]):
    '''
    The function takes an image and performs a set of random affine transformation.
    img:original images
    ang_range:angular range of the rotation [-15,+15] deg for example
    scale_range: [0.8,1.2]
    shear_range:[10,-10]
    translation_range:[-2,2]
    '''
    img_height,img_width,img_depth = img.shape
    # Generate random parameter values
    angle_value = np.random.uniform(low=angle_range[0],high=angle_range[1],size=None)
    scaleX = np.random.uniform(low=scale_range[0],high=scale_range[1],size=None)
    scaleY = np.random.uniform(low=scale_range[0],high=scale_range[1],size=None)
    translationX = np.random.randint(low=translation_range[0],high=translation_range[1]+1,size=None)
    translationY = np.random.randint(low=translation_range[0],high=translation_range[1]+1,size=None)
    
    center_shift = np.array([img_height,img_width])/2. - 0.5
    transform_center = transf.SimilarityTransform(translation=-center_shift)
    transform_uncenter = transf.SimilarityTransform(translation=center_shift)
    
    transform_aug = transf.AffineTransform(rotation=np.deg2rad(angle_value),
                                          scale=(1/scaleY,1/scaleX),
                                          translation = (translationY,translationX))
    #Image transformation : includes rotation ,shear,translation,zoom
    full_tranform = transform_center + transform_aug + transform_uncenter
    new_img = transf.warp(img,full_tranform,preserve_range=True)
    
    return new_img.astype('uint8')

def data_augmentation(X_dataset,y_dataset,augm_nbr,keep_dist=True):
    '''
    X_dataset:image dataset to augment
    y_dataset:label dataset
    keep_dist - True:keep class distribution of original dataset,
                False:balance dataset
    augm_param - is the augmentation parameter
                if keep_dist is True,increase the dataset by the factor 'augm_nbr' (2x,5x or 10x...)
                if keep_dist is False,make all classes have same number of images:'augm_nbr'(2500,3000 or 4000 imgs)
    '''
    X_train_dtype = X_train
    n_classes = len(np.unique(y_dataset))
    _,img_height,img_width,img_depth = X_dataset.shape
    class_freq = get_count_imgs_per_class(y_train)
    
    if keep_dist:
        extra_imgs_per_class = np.array([augm_nbr*x for x in get_count_imgs_per_class(y_dataset)])
    else:
        assert (augm_nbr>np.argmax(class_freq)),'augm_nbr must be larger than the height class count'
        extra_imgs_per_class = augm_nbr - get_count_imgs_per_class(y_dataset)
    
    total_extra_imgs = np.sum(extra_imgs_per_class)
    
    #if extra data is needed->run the dataaumentation op
    if total_extra_imgs > 0:
        X_extra = np.zeros((int(total_extra_imgs),img_height,img_width,img_depth),dtype=X_train.dtype)
        y_extra = np.zeros(int(total_extra_imgs))
        start_idx = 0
        print('start data augmentation.....')
        for this_class in range(n_classes):
            print('\t Class {}|Number of extra imgs{}'.format(this_class,int(extra_imgs_per_class[this_class])))
            n_extra_imgs = extra_imgs_per_class[this_class]
            end_idx = start_idx + n_extra_imgs
            
            if n_extra_imgs > 0:
                #get ids of all images belonging to this_class
                all_imgs_id = np.argwhere(y_dataset==this_class)
                new_imgs_x = np.zeros((int(n_extra_imgs),img_height,img_width,img_depth))
                
                for k in range(int(n_extra_imgs)):
                    #randomly pick an original image belonging to this class
                    rand_id = np.random.choice(all_imgs_id[0],size=None,replace=True)
                    rand_img = X_train[rand_id]
                    #Transform image
                    new_img = random_transform(rand_img)
                    new_imgs_x[k,:,:,:] = new_img
                #update tensors with new images and associated labels
                X_extra[int(start_idx):int(end_idx)] = new_imgs_x
                y_extra[int(start_idx):int(end_idx)] = np.ones((int(n_extra_imgs),))*this_class
                start_idx = end_idx
        return [X_extra,y_extra]
    else:
        return [None,None]
# shuffle train dataset before split
X_train,y_train = shuffle(X_train,y_train)
_,IMG_HEIGHT,IMG_WIDTH,IMG_DEPTH = X_train.shape

X_train,X_validation,y_train,y_validation = train_test_split(X_train,y_train,test_size=0.2,random_state=SEED)
print('Train set size:{}|Validation set size:{}\n'.format(X_train.shape[0],X_validation.shape[0]))

X_extra,y_extra = data_augmentation(X_train,y_train,augm_nbr=4000,keep_dist=False)
```

    Train set size:27839|Validation set size:6960
    
    start data augmentation.....
    	 Class 0|Number of extra imgs3855
    	 Class 1|Number of extra imgs2407
    	 Class 2|Number of extra imgs2411
    	 Class 3|Number of extra imgs2985
    	 Class 4|Number of extra imgs2577
    	 Class 5|Number of extra imgs2677
    	 Class 6|Number of extra imgs3715
    	 Class 7|Number of extra imgs2965
    	 Class 8|Number of extra imgs2987
    	 Class 9|Number of extra imgs2953
    	 Class 10|Number of extra imgs2570
    	 Class 11|Number of extra imgs3047
    	 Class 12|Number of extra imgs2481
    	 Class 13|Number of extra imgs2477
    	 Class 14|Number of extra imgs3444
    	 Class 15|Number of extra imgs3572
    	 Class 16|Number of extra imgs3711
    	 Class 17|Number of extra imgs3206
    	 Class 18|Number of extra imgs3163
    	 Class 19|Number of extra imgs3861
    	 Class 20|Number of extra imgs3770
    	 Class 21|Number of extra imgs3786
    	 Class 22|Number of extra imgs3739
    	 Class 23|Number of extra imgs3631
    	 Class 24|Number of extra imgs3800
    	 Class 25|Number of extra imgs2922
    	 Class 26|Number of extra imgs3566
    	 Class 27|Number of extra imgs3828
    	 Class 28|Number of extra imgs3615
    	 Class 29|Number of extra imgs3812
    	 Class 30|Number of extra imgs3684
    	 Class 31|Number of extra imgs3453
    	 Class 32|Number of extra imgs3850
    	 Class 33|Number of extra imgs3511
    	 Class 34|Number of extra imgs3704
    	 Class 35|Number of extra imgs3132
    	 Class 36|Number of extra imgs3733
    	 Class 37|Number of extra imgs3853
    	 Class 38|Number of extra imgs2518
    	 Class 39|Number of extra imgs3783
    	 Class 40|Number of extra imgs3753
    	 Class 41|Number of extra imgs3828
    	 Class 42|Number of extra imgs3826



```python
# Visualize 20 examples picked randomly from train dataset
ls = [random.randint(0,len(y_extra)) for i in range(20)]
visualize_random_images(list_imgs=ls,X_dataset=X_extra,y_dataset=y_extra)
```


![png](https://img-blog.csdn.net/20180514102955180?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



```python
if X_extra is not None:
    X_train = np.concatenate((X_train,X_extra.astype('uint8')),axis=0)
    y_train = np.concatenate((y_train,y_extra),axis=0)
    del X_extra,y_extra
```

# visualization after data augmentation
>* Display 20 random images
>* show frequency of each class


```python
ls = [random.randint(0,len(y_train)) for i in range(20)]
visualize_random_images(list_imgs=ls,X_dataset=X_train,y_dataset=y_train)

print('*** Train dataset after augmentation')
print('\t Total Number of images in Train dataset:{}'.format(X_train.shape[0]))

plt.bar(np.arange(n_classes),get_count_imgs_per_class(y_train),align='center')
plt.xlabel('class')
plt.ylabel('Frequency')
plt.xlim([-1,43])
plt.show()

print('*** Validation dataset')
plt.bar(np.arange(n_classes),get_count_imgs_per_class(y_validation),align='center')
plt.xlabel('class')
plt.ylabel('Frequency')
plt.xlim([-1,43])
plt.show()
```


![png](https://img-blog.csdn.net/20180514103010504?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


    *** Train dataset after augmentation
    	 Total Number of images in Train dataset:172000



![png](https://img-blog.csdn.net/20180514103043464?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


    *** Validation dataset



![png](https://img-blog.csdn.net/20180514103035937?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



```python
def preprocessed(dataset):
    n_imgs,img_height,img_width,_ = dataset.shape
    processed_dataset = np.zeros((n_imgs,img_height,img_width,1))
    for idx in range(len(dataset)):
        img = dataset[idx]
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        
        processed_dataset[idx,:,:,0] = gray/255. - 0.5
    return processed_dataset
```

# 设计并测试模型架构
接下来就是我们的重头戏了。
我们需要设计并实现一个深度学习模型并用来学习识别交通信号。
在这个过程中，我们需要思考并考虑以下几点内容：
>* 神经网络架构
>* 如何应用前面实现好的预处理技术
>* 样本不均匀带来的问题
>* 生成虚假数据
具体的设计思路，大家可以参考LeCun于2011年发表的文章：[Traffic Sign Recognition with Multi-Scale Convolutional Networks](https://scholar.google.com/scholar_url?url=http://ieeexplore.ieee.org/abstract/document/6033589/&hl=zh-CN&sa=T&oi=gsb&ct=res&cd=0&ei=3rrqWtr1IsSyyASpkrLgDA&scisig=AAGBfm1qTtsdXG41K740eWaSoOr0-BKhoQ)

这里我们设计一个简单的卷积神经网络，它由两大部分组成：
**1:**卷积层
**2:**全连接层
具体架构图如下所示：



```python
# Variables Initialization function and Operation
def weight_variable(shape,mean,stddev,name,seed=SEED):
    init = tf.truncated_normal(shape,mean=mean,stddev=stddev,seed=SEED)
    return tf.Variable(init,name=name)

def bias_variable(shape,init_value,name):
    init = tf.constant(init_value,shape=shape)
    return tf.Variable(init,name=name)

def conv2d(x,W,strides,padding,name):
    return tf.nn.conv2d(x,W,strides=strides,padding=padding,name=name)

def max_2x2_pool(x,padding,name):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding=padding,name=name)
```


```python
#weights and biases
#parameters

IMG_DEPTH = 1
mu =0
sigma = 0.05
bias_init = 0.05

weights ={  
    'W_conv1': weight_variable([3, 3, IMG_DEPTH, 80], mean=mu, stddev=sigma, name='W_conv1'),
    'W_conv2': weight_variable([3, 3, 80, 120], mean=mu, stddev=sigma, name='W_conv2'),
    'W_conv3': weight_variable([4, 4, 120, 180], mean=mu, stddev=sigma, name='W_conv3'),
    'W_conv4': weight_variable([3, 3, 180, 200], mean=mu, stddev=sigma, name='W_conv4'),
    'W_conv5': weight_variable([3, 3, 200, 200], mean=mu, stddev=sigma, name='W_conv5'),
    'W_fc1': weight_variable([800, 80], mean=mu, stddev=sigma, name='W_fc1'),
    'W_fc2': weight_variable([80, 80], mean=mu, stddev=sigma, name='W_fc2'),
    'W_fc3': weight_variable([80, 43], mean=mu, stddev=sigma, name='W_fc3'),
}
biases = {
    'b_conv1': bias_variable(shape=[80], init_value=bias_init, name='b_conv1'),
    'b_conv2': bias_variable(shape=[120], init_value=bias_init, name='b_conv2'),
    'b_conv3': bias_variable(shape=[180], init_value=bias_init, name='b_conv3'),
    'b_conv4': bias_variable(shape=[200], init_value=bias_init, name='b_conv4'),
    'b_conv5': bias_variable(shape=[200], init_value=bias_init, name='b_conv5'),
    'b_fc1': bias_variable([80], init_value=bias_init, name='b_fc1'),
    'b_fc2': bias_variable([80], init_value=bias_init, name='b_fc2'),
    'b_fc3': bias_variable([43], init_value=bias_init, name='b_fc3'),
}

```


```python
def traffic_model(x,keep_prob,keep_p_conv,weights,biases):
    '''
    ConvNet model for Traffic sign classifier
    x - input image is tensor of shape(n_imgs,img_height,img_width,img_depth)
    keep_prob - hyper parameter of the dropout operation
    weights - dictionary of the weights for convolution layers and fully connected layers
    biases dictionary of the biases for convolutional layers and fully connected layers
    '''
    # Convolutional block 1
    conv1 = conv2d(x, weights['W_conv1'], strides=[1,1,1,1], padding='VALID', name='conv1_op')
    conv1_act = tf.nn.relu(conv1 + biases['b_conv1'], name='conv1_act')
    conv1_drop = tf.nn.dropout(conv1_act, keep_prob=k_p_conv, name='conv1_drop')
    conv2 = conv2d(conv1_drop, weights['W_conv2'], strides=[1,1,1,1], padding='SAME', name='conv2_op')
    conv2_act = tf.nn.relu(conv2 + biases['b_conv2'], name='conv2_act')
    conv2_pool = max_2x2_pool(conv2_act, padding='VALID', name='conv2_pool')
    pool2_drop = tf.nn.dropout(conv2_pool, keep_prob=k_p_conv, name='conv2_drop')
    
    #Convolution block 2
    conv3 = conv2d(pool2_drop, weights['W_conv3'], strides=[1,1,1,1], padding='VALID', name='conv3_op')
    conv3_act = tf.nn.relu(conv3 + biases['b_conv3'], name='conv3_act')
    conv3_drop = tf.nn.dropout(conv3_act, keep_prob=k_p_conv, name='conv3_drop')
    conv4 = conv2d(conv3_drop, weights['W_conv4'], strides=[1,1,1,1], padding='SAME', name='conv4_op')
    conv4_act = tf.nn.relu(conv4 + biases['b_conv4'], name='conv4_act')
    conv4_pool = max_2x2_pool(conv4_act, padding='VALID', name='conv4_pool')
    conv4_drop = tf.nn.dropout(conv4_pool, keep_prob, name='conv4_drop')
    
    conv5 = conv2d(conv4_drop, weights['W_conv5'], strides=[1,1,1,1], padding='VALID', name='conv5_op')
    conv5_act = tf.nn.relu(conv5 + biases['b_conv5'], name='conv5_act')
    conv5_pool = max_2x2_pool(conv5_act, padding='VALID', name='conv5_pool')
    conv5_drop = tf.nn.dropout(conv5_pool, keep_prob, name='conv5_drop')
    
    #Fully connected layers
    fc0 = flatten(conv5_drop)
    fc1 = tf.nn.relu( tf.matmul( fc0, weights['W_fc1'] ) + biases['b_fc1'], name='fc1' )
    fc1_drop = tf.nn.dropout(fc1, keep_prob, name='fc1_drop')
    fc2 = tf.nn.relu( tf.matmul( fc1_drop, weights['W_fc2'] ) + biases['b_fc2'], name='fc2' )
    fc2_drop = tf.nn.dropout(fc2, keep_prob, name='fc2_drop')
    logits = tf.add(tf.matmul(fc2_drop, weights['W_fc3']),biases['b_fc3'], name='logits')  
    
    return [weights, logits]
```


```python
# Train your model here
x = tf.placeholder(tf.float32,(None,IMG_HEIGHT,IMG_WIDTH,IMG_DEPTH),name='x')
y = tf.placeholder(tf.int32,(None),name='y')
keep_prob = tf.placeholder(tf.float32,name='keep_prob')
k_p_conv = tf.placeholder(tf.float32,name='k_p_conv')
one_hot_y = tf.one_hot(y,n_classes)
rate = tf.placeholder(tf.float32,name='rate')

weights,logits = traffic_model(x,keep_prob,k_p_conv,weights,biases)
softmax_operation = tf.nn.softmax(logits)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits,labels = one_hot_y)
beta = 0.0001
loss_reg = beta*(tf.nn.l2_loss(weights['W_fc1'])+tf.nn.l2_loss(weights['W_fc2'])+tf.nn.l2_loss(weights['W_fc3']))
loss = tf.reduce_mean(cross_entropy)+loss_reg

optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss)
```


```python
correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(one_hot_y,1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

def evaluate(X_data,y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    total_l = 0
    for offset in range(0,num_examples,BATCH_SIZE):
        batch_x,batch_y = X_data[offset:offset+BATCH_SIZE],y_data[offset:offset+BATCH_SIZE]
        accuracy,l = sess.run([accuracy_operation,loss],feed_dict={x:batch_x,y:batch_y,k_p_conv:1,keep_prob:1})
        total_accuracy+=(accuracy*len(batch_x))
        total_l +=l*len(batch_x)
    return [total_accuracy/num_examples,total_l/num_examples]
```


```python
'''
histogram equalzier turn off
EPOCHs=100
l_rate decreases from 0.001 to l_rate/5 at 30 EPOCH and 50 EPOCHS
Keep same class distribution as original dataset:augmentation=6X
No keep prob for conv
'''
EPOCHS = 150
BATCH_SIZE = 200
model_nbr = 'ora'
```


```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    print('Training... \n')
    summary_train = []
    l_rate = 0.001
    keep_rate = 0.5
    kp_conv  = 0.6
    print('Pre-processing X_train...')
    X_train_prep = preprocessed(X_train).reshape(-1,IMG_HEIGHT,IMG_WIDTH,IMG_DEPTH)
    X_val_prep = preprocessed(X_validation).reshape(-1,IMG_HEIGHT,IMG_WIDTH,IMG_DEPTH)
    print('X_train preprocessed dataset size:{}|data type:{}'.format(X_train_prep.shape,X_train_prep.dtype))
    print('End preprocessing X_train...')
    
    #map for reduction of l_rate at different EPOCHS(first elt)
    
    for i in range(EPOCHS):
        #scheme to decrease learning rate by step
        if i >=40:
            l_rate = 0.0001
        
        X_train_prep,y_train = shuffle(X_train_prep,y_train)
        
        for offset in range(0,num_examples,BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x,batch_y = X_train_prep[offset:end],y_train[offset:end]
            
            sess.run(training_operation,feed_dict={x:batch_x,y:batch_y,keep_prob:keep_rate,\
                                                  k_p_conv:kp_conv,rate:l_rate})
        
        train_accuracy,train_loss = evaluate(X_train_prep,y_train)
        
        validation_accuracy,validation_loss = evaluate(X_val_prep,y_validation)
        
        print('EPOCH{}...'.format(i+1))
        print('Train accuracy:{:.4f}|Validation Accuracy={:.4f}'.format(train_accuracy,validation_accuracy))
        print('Train loss:{:.5f}|Validation loss = {:.5f}\n'.format(train_loss,validation_loss))
        summary_train.append([i+1,train_accuracy,validation_accuracy,train_loss,validation_loss])
        
    summary_train = np.array(summary_train)
    np.save('summary_train_'+model_nbr+'.npy',summary_train)
    try:
        saver
    except NameError:
        saver = tf.train.Saver()
    saver.save(sess,save_path='./traffic_model'+model_nbr)
    print('Model saved')
    '''
    ##Plot loss
    fig,ax = plt.subplots(1,3,figsize=(15,3))
    plt.subplots_adjust(wspace=.2)
    # set font size tick parameters ,x/y labels
    for i in range(len(ax)):
        ax[i].tick_params(axis='x',labelsize=12)
        ax[i].tick_params(axis='y',labelsize=12)
        ax[i].xaxis.label.set_fontsize(12)
        ax[i].yaxis.label.set_fontsize(12)
    marker_size = 8
    ax[0].plot(summary_train[:,0],summary_train[:,1],'b-o',markersize=marker_size,label='Train')
    ax[0].plot(summary_train[:,0],summary_train[:,2],'r-o',markersize=marker_size,label='Validation')
    ax[0].set_xlabel('EPOCH')
    ax[0].set_ylabel('ACCURACY')
    ax[1].semilogy(summary_train[:,0],summary_train[:,3],'b-o',markersize=marker_size,label='Train')
    ax[1].semilogy(summary_train[:,0],summary_train[:,4],'r-o',markersize=marker_size,label='Validation')
    
    ax[1].set_xlabel('EPOCH')
    ax[1].set_ylabel('LOSS')
    ax[2].semilogy(summary_train[:,0],summary_train[:,3]/summary_train[:,4],'k-o',markersize=marker_size,label='Train')
    
    ax[2].set_xlabel('EPOCH')
    ax[2].set_ylabel('LOSS RATIO TRAIN/VALID')
    
    plt.show()
    '''
```

    Training... 
    
    Pre-processing X_train...
    X_train preprocessed dataset size:(172000, 32, 32, 1)|data type:float64
    End preprocessing X_train...
    EPOCH1...
    Train accuracy:0.7694|Validation Accuracy=0.1568
    Train loss:0.85597|Validation loss = 3.20545
    
    EPOCH2...
    Train accuracy:0.8717|Validation Accuracy=0.3497

    ......
    
    EPOCH149...
    Train accuracy:1.0000|Validation Accuracy=0.9990
    Train loss:0.00731|Validation loss = 0.01181
    
    EPOCH150...
    Train accuracy:1.0000|Validation Accuracy=0.9991
    Train loss:0.00728|Validation loss = 0.01087
    
    Model saved


# Test
模型训练完了，我们可以从网上选取一些图片来测试下咱们训练的分类器是否能正确识别交通标志。
# Implementation
接下来我们加载前面训练好的模型，并预测结果


```python
with tf.Session() as sess:
    
    loader = tf.train.import_meta_graph('traffic_model'+model_nbr+'.meta')
    sess.run(tf.global_variables_initializer())
    loader.restore(sess,tf.train.latest_checkpoint(checkpoint_dir='./'))
    X_test_prep = preprocessed(X_test).reshape(-1,IMG_HEIGHT,IMG_WIDTH,1)
    test_accuracy,_=evaluate(X_test_prep,y_test)
    
    #select 20 random images
    ls = [random.randint(0,len(y_test)) for i in range(20)]
    X_test_select = np.zeros((20,IMG_HEIGHT,IMG_WIDTH,1))
    y_test_select = np.zeros((20,1))
    
    for i in range(len(ls)):
        X_test_select[i] = X_test_prep[ls[i]]
        y_test_select = y_test[ls[i]]
    
    test_pred_proba = sess.run(softmax_operation,feed_dict={x:X_test_select,k_p_conv:1,keep_prob:1})
    prediction_test = np.argmax(test_pred_proba,1)
    print('Test Accuracy={:.4f}'.format(test_accuracy))
```

    INFO:tensorflow:Restoring parameters from ./traffic_modelora
    Test Accuracy=0.9825



```python
#Visualization
#random select images and their predicted labels
_,ax = plt.subplots(len(ls)//5,5,figsize=(6,5))
row,col = 0,0
for i ,idx in enumerate(ls):
    img = X_test[idx]
    ax[row,col].imshow(img,cmap='gray')
    annot = 'pred:'+str(int(prediction_test[i]))+'|True:'+str(y_test[idx])
    ax[row,col].annotate(annot,xy=(0,5),color='black',fontsize='7',bbox=dict(boxstyle='round',fc='0.8'))
    
    ax[row,col].axis('off')
    col+=1
    if col == 5:
        row,col = row+1,0
plt.show()
```


![png](https://img-blog.csdn.net/20180514103246542?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



```python
import os

IMG_HEIGHT = 32
IMG_WIDTH = 32

def get_list_files(my_dir,f_ext):
    list_f = []
    for file in os.listdir(my_dir):
        if file.endswith('.'+f_ext):
            list_f.append(file)
    return list_f

my_dir = 'extra'

file_list = get_list_files(my_dir,'png')

X_extra = np.zeros((len(file_list),IMG_HEIGHT,IMG_WIDTH,3),dtype='uint8')

for idx,file in enumerate(file_list):
    img = cv2.imread(my_dir+'/'+file)
    img = cv2.resize(img,(32,32))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    X_extra[idx] = img

print('Extra dataset size:{}|Datatype:{}'.format(X_extra.shape,X_extra.dtype))
#Data pre-processing
X_extra_prep = preprocessed(X_extra).reshape(-1,IMG_HEIGHT,IMG_WIDTH,1)
print('Preprocessed Extra dataset size:{}|Dtatype:{}'.format(X_extra_prep.shape,X_extra_prep.dtype))
```

    Extra dataset size:(10, 32, 32, 3)|Datatype:uint8
    Preprocessed Extra dataset size:(10, 32, 32, 1)|Dtatype:float64



```python
#Visualize images:original(left) and after pre-processing(right)

#initialize subplots
_,ax = plt.subplots(len(file_list),2,figsize=(4,8))
col_plot = 0
print('Original (left) and pre-processed(right) iamges')
for i in range(len(X_extra)):
    img = X_extra[i]
    ax[i,col_plot].imshow(img)
    ax[i,col_plot].annotate(file_list[i],xy=(31,5),color='black',fontsize='10')
    ax[i,col_plot].axis('off')
    col_plot +=1
    ax[i,col_plot].imshow(X_extra_prep[i,:,:,0],cmap='gray')
    ax[i,col_plot].axis('off')
    col_plot = 0
plt.show()
```

    Original (left) and pre-processed(right) iamges



![png](https://img-blog.csdn.net/20180514103258382?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



```python
##inference
feed_dict = {x:X_extra_prep,keep_prob:1,k_p_conv:1}
k_top = 5
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loader = tf.train.import_meta_graph('traffic_model'+model_nbr+'.meta')
    loader.restore(sess,tf.train.latest_checkpoint('./'))
    pred_proba = sess.run(softmax_operation,feed_dict=feed_dict)
    prediction = np.argmax(pred_proba,1)
    # top 5 probabilities
    top_k_values = tf.nn.top_k(softmax_operation,k_top)
    top_k_proba = sess.run([softmax_operation,top_k_values],feed_dict=feed_dict)

#Visualize image with predicted label
print('Prediction on extra data')
for i in range(len(X_extra)):
    plt.figure(figsize=(1,1))
    img = X_extra[i]
    plt.imshow(img)
    plt.title(sign_names[prediction[i]][1],fontsize=10)
    plt.axis('off')
    plt.show()
```

    INFO:tensorflow:Restoring parameters from ./traffic_modelora
    Prediction on extra data



![png](https://img-blog.csdn.net/20180514103310925?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



![png](https://img-blog.csdn.net/2018051410331875?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



![png](https://img-blog.csdn.net/20180514103331450?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



![png](https://img-blog.csdn.net/20180514103339610?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



![png](https://img-blog.csdn.net/2018051410334947?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



![png](https://img-blog.csdn.net/20180514103356933?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



![png](https://img-blog.csdn.net/2018051410340813?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70![png](https://img-blog.csdn.net/20180514103425292?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![png\](output_27_9.png)](https://img-blog.csdn.net/20180514103446617?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



![png](https://img-blog.csdn.net/20180514103500319?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


可以看的出来上面的预测结果有的是不正确的，这与我们前面模型训练有关。


```python
###TOP 5 Probabilities 
_,ax = plt.subplots(len(file_list),2,figsize=(4,8))
col_plot = 0
for i in range(len(X_extra)):
    img = X_extra[i]
    ax[i,col_plot].imshow(img)
    ax[i,col_plot].axis('off')
    col_plot +=1
    ax[i,col_plot].barh(-np.arange(k_top),top_k_proba[1][0][i],align='center')
    
    #annotation
    for k in range(k_top):
        text_pos = [top_k_proba[1][0][i][k]+.1,-(k+0.4)]
        ax[i,col_plot].text(text_pos[0],text_pos[1],sign_names[top_k_proba[1][1][i][k]][1],fontsize=8)
    ax[i,col_plot].axis('off')
    col_plot = 0
plt.show()                                                     
```


![png](https://img-blog.csdn.net/20180514103509549?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


# Multi-Scale Convolutional Networks
前面我们采用的网络结构是传统经典的卷积网络结构,接下来我们实现一个multi-scaled network from lecun
这里我只实现了代码，没有运行结果，因为运行起来时间比较长。

```python
#weights
weights ={  
    'W_conv1': weight_variable([3, 3, IMG_DEPTH, 80], mean=mu, stddev=sigma, name='W_conv1'),
    'W_conv2': weight_variable([3, 3, 80, 120], mean=mu, stddev=sigma, name='W_conv2'),
    'W_conv3': weight_variable([4, 4, 120, 180], mean=mu, stddev=sigma, name='W_conv3'),
    'W_conv4': weight_variable([3, 3, 180, 200], mean=mu, stddev=sigma, name='W_conv4'),
    'W_conv5': weight_variable([3, 3, 200, 200], mean=mu, stddev=sigma, name='W_conv5'),
    'W_fc1': weight_variable([8000, 80], mean=mu, stddev=sigma, name='W_fc1'),
    'W_fc2': weight_variable([80, 80], mean=mu, stddev=sigma, name='W_fc2'),
    'W_fc3': weight_variable([80, 43], mean=mu, stddev=sigma, name='W_fc3'),
}
def traffic_model_Lecun(x,keep_prob,keep_p_conv,weights,biases):
    '''
    ConvNet model for Traffic sign classifier
    x - input image is tensor of shape(n_imgs,img_height,img_width,img_depth)
    keep_prob - hyper parameter of the dropout operation
    weights - dictionary of the weights for convolution layers and fully connected layers
    biases dictionary of the biases for convolutional layers and fully connected layers
    '''
    # Convolutional block 1
    conv1 = conv2d(x, weights['W_conv1'], strides=[1,1,1,1], padding='VALID', name='conv1_op')
    conv1_act = tf.nn.relu(conv1 + biases['b_conv1'], name='conv1_act')
    conv1_drop = tf.nn.dropout(conv1_act, keep_prob=k_p_conv, name='conv1_drop')
    conv2 = conv2d(conv1_drop, weights['W_conv2'], strides=[1,1,1,1], padding='SAME', name='conv2_op')
    conv2_act = tf.nn.relu(conv2 + biases['b_conv2'], name='conv2_act')
    conv2_pool = max_2x2_pool(conv2_act, padding='VALID', name='conv2_pool')
    pool2_drop = tf.nn.dropout(conv2_pool, keep_prob=k_p_conv, name='conv2_drop')
    
    #Convolution block 2
    conv3 = conv2d(pool2_drop, weights['W_conv3'], strides=[1,1,1,1], padding='VALID', name='conv3_op')
    conv3_act = tf.nn.relu(conv3 + biases['b_conv3'], name='conv3_act')
    conv3_drop = tf.nn.dropout(conv3_act, keep_prob=k_p_conv, name='conv3_drop')
    conv4 = conv2d(conv3_drop, weights['W_conv4'], strides=[1,1,1,1], padding='SAME', name='conv4_op')
    conv4_act = tf.nn.relu(conv4 + biases['b_conv4'], name='conv4_act')
    conv4_pool = max_2x2_pool(conv4_act, padding='VALID', name='conv4_pool')
    conv4_drop = tf.nn.dropout(conv4_pool, keep_prob, name='conv4_drop')
    #Convolution block 3
    conv5 = conv2d(conv4_drop, weights['W_conv5'], strides=[1,1,1,1], padding='VALID', name='conv5_op')
    conv5_act = tf.nn.relu(conv5 + biases['b_conv5'], name='conv5_act')
    conv5_pool = max_2x2_pool(conv5_act, padding='VALID', name='conv5_pool')
    conv5_drop = tf.nn.dropout(conv5_pool, keep_prob, name='conv5_drop')
    # Flatten the out put convolution block 2
    fc_ = flatten(conv4_drop)
    #Fully connected layers
    fc0 = flatten(conv5_drop)
    fc = tf.concat([fc_,fc0],1)
    print('fc shape:',fc.get_shape())
    fc1 = tf.nn.relu( tf.matmul( fc, weights['W_fc1'] ) + biases['b_fc1'], name='fc1' )
    fc1_drop = tf.nn.dropout(fc1, keep_prob, name='fc1_drop')
    fc2 = tf.nn.relu( tf.matmul( fc1_drop, weights['W_fc2'] ) + biases['b_fc2'], name='fc2' )
    fc2_drop = tf.nn.dropout(fc2, keep_prob, name='fc2_drop')
    logits = tf.add(tf.matmul(fc2_drop, weights['W_fc3']),biases['b_fc3'], name='logits')  
    
    return [weights, logits]
```

# 论文复现
下面的代码是对Lecun 2011年发表的[Traffic Sign Recognition with Multi-Scale Convolutional Networks](https://download.csdn.net/download/u010665216/10412418)的复现。

# 说明
本篇博客采用的卷积神经网络模型是Lecun于2011年发表的论文：
[Traffic Sign Recognition with Multi-Scale Convolutional Networks](https://scholar.google.com/scholar_url?url=http://ieeexplore.ieee.org/abstract/document/6033589/&hl=zh-CN&sa=T&oi=gsb&ct=res&cd=0&ei=3rrqWtr1IsSyyASpkrLgDA&scisig=AAGBfm1qTtsdXG41K740eWaSoOr0-BKhoQ)


```python
# Load data
import pickle
training_file = 'data/train.p'
testing_file = 'data/test.p'

with open(training_file,mode='rb') as f:
    train = pickle.load(f)
with open(testing_file,mode='rb') as f:
    test = pickle.load(f)

X_train,y_train = train['features'],train['labels']
X_test,y_test = test['features'],test['labels']

print('X_train shape:',X_train.shape)
print('y_train shape:',y_train.shape)
print('X_test shape:',X_test.shape)
print('y_test shape:',y_test.shape)
```

    X_train shape: (34799, 32, 32, 3)
    y_train shape: (34799,)
    X_test shape: (12630, 32, 32, 3)
    y_test shape: (12630,)



```python
import csv
import numpy as np
n_train = len(X_train)
n_test = len(X_test)

_,IMG_HEIGHT,IMG_WIDTH,IMG_DEPTH = X_train.shape
image_shape = (IMG_HEIGHT,IMG_WIDTH,IMG_DEPTH)

with open('data/signnames.csv','r') as sign_name:
    reader = csv.reader(sign_name)
    sign_names = list(reader)

sign_names = sign_names[1::]
NUM_CLASSES = len(sign_names)
print('Total number of classes:{}'.format(NUM_CLASSES))

n_classes = len(np.unique(y_train))
assert (NUM_CLASSES== n_classes) ,'1 or more class(es) not represented in training set'

n_test = len(y_test)

print('Number of training examples =',n_train)
print('Number of testing examples =',n_test)
print('Image data shape=',image_shape)
print('Number of classes =',n_classes)
```

    Total number of classes:43
    Number of training examples = 34799
    Number of testing examples = 12630
    Image data shape= (32, 32, 3)
    Number of classes = 43



```python
import matplotlib.pyplot as plt
import random
%matplotlib inline

# show image of 10 random data points
fig,axs = plt.subplots(2,5,figsize=(15,6))
fig.subplots_adjust(hspace=.2,wspace=.001)
axs = axs.ravel()
for i in range(10):
    index = random.randint(0,len(X_train))
    image = X_train[index]
    axs[i].axis('off')
    axs[i].imshow(image)
    axs[i].set_title(y_train[index])
```


![png](https://img-blog.csdn.net/20180514103752456?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



```python
hist,bins = np.histogram(y_train,bins=n_classes)
print(bins)
width = 0.7*(bins[1]-bins[0])
center = (bins[:-1]+bins[1:])/2
plt.bar(center,hist,align='center',width=width)
plt.show()
```

    [ 0.          0.97674419  1.95348837  2.93023256  3.90697674  4.88372093
      5.86046512  6.8372093   7.81395349  8.79069767  9.76744186 10.74418605
     11.72093023 12.69767442 13.6744186  14.65116279 15.62790698 16.60465116
     17.58139535 18.55813953 19.53488372 20.51162791 21.48837209 22.46511628
     23.44186047 24.41860465 25.39534884 26.37209302 27.34883721 28.3255814
     29.30232558 30.27906977 31.25581395 32.23255814 33.20930233 34.18604651
     35.1627907  36.13953488 37.11627907 38.09302326 39.06976744 40.04651163
     41.02325581 42.        ]



![png](https://img-blog.csdn.net/20180514103802201?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



```python
# Design and Test a model Architecture
X_train_rgb = X_train
X_train_gry = np.sum(X_train/3,axis=3,keepdims=True)

X_test_rgb = X_test
X_test_gry = np.sum(X_test/3,axis=3,keepdims=True)

print('RGB shape:',X_train_rgb.shape)
print('Graysacle shape:',X_train_gry.shape)
```

    RGB shape: (34799, 32, 32, 3)
    Graysacle shape: (34799, 32, 32, 1)



```python
X_train = X_train_gry
X_test = X_test_gry
# Visualize rgb vs grayscale
n_rows = 8
n_cols = 10
offset = 9000
fig,axs = plt.subplots(n_rows,n_cols,figsize=(18,14))
fig.subplots_adjust(hspace=.1,wspace=.001)
axs = axs.ravel()
for j in range(0,n_rows,2):
    for i in range(n_cols):
        index = i + j*n_cols
        image = X_train_rgb[index+offset]
        axs[index].axis('off')
        axs[index].imshow(image)
    for i in range(n_cols):
        index = i + j*n_cols +n_cols
        image = X_train_gry[index + offset-n_cols].squeeze()
        axs[index].axis('off')
        axs[index].imshow(image,cmap='gray')
```


![png](https://img-blog.csdn.net/20180514103811320?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



```python
X_train[0][0][0]
```




    array([25.66666667])




```python
# Normalize the train and test datasets to (-1,1)
X_train_normalized = (X_train -128)/128
X_test_normalized = (X_test - 128)/128

```

# Preprocess


```python
import cv2
def random_translate(img):
    rows,cols,_ = img.shape
    
    #allow translation up to px pixels in x and y directions
    px = 2
    dx,dy = np.random.randint(-px,px,2)
    
    M = np.float32([[1,0,dx],[0,1,dy]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    
    dst = dst[:,:,np.newaxis]
    return dst

test_img = X_train_normalized[22222]

test_dst = random_translate(test_img)

fig,axs = plt.subplots(1,2,figsize=(10,3))

axs[0].axis('off')
axs[0].imshow(test_img.squeeze(),cmap='gray')
axs[0].set_title('original')

axs[1].axis('off')
axs[1].imshow(test_dst.squeeze(),cmap='gray')
axs[1].set_title('translated')

print('shape in/out:',test_img.shape,test_dst.shape)
```

    shape in/out: (32, 32, 1) (32, 32, 1)



![png](https://img-blog.csdn.net/20180514103823539?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



```python
def random_scaling(img):
    rows,cols,_ = img.shape
    
    #transform limits
    px = np.random.randint(-2,2)
    
    # ending locations
    pts1 = np.float32([[px,px],[rows-px,px],[px,cols-px],[rows-px,cols-px]])
    
    #starting locations (4 corners)
    pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])
    
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(rows,cols))
    dst = dst[:,:,np.newaxis]
    
    return dst
test_dst = random_scaling(test_img)
fig,axs = plt.subplots(1,2,figsize=(10,3))

#print(test_dst.shape)
#print(test_dst.squeeze().shape)
axs[0].axis('off')
axs[0].imshow(test_img.squeeze(),cmap='gray')
axs[0].set_title('original')

axs[1].axis('off')
axs[1].imshow(test_dst.squeeze(),cmap='gray')
axs[1].set_title('scaled')

print('shape in/out:',test_img.shape,test_dst.shape)
```

    shape in/out: (32, 32, 1) (32, 32, 1)



![png](https://img-blog.csdn.net/20180514103832155?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



```python
def random_warp(img):
    rows,cols,_ = img.shape
    
    # random scaling cofficients
    rndx = np.random.rand(3) - 0.5
    rndx *=cols*0.06
    rndy = np.random.rand(3) - 0.5
    rndy *=rows*0.06
    
    # 3 starting points for transform,1/4 way from edges
    x1 = cols/4
    x2 = 3*cols/4
    y1 = rows/4
    y2 = 3*rows/4
    
    pts1 = np.float32([[y1,x1],
                      [y2,x1],
                      [y1,x2]])
    pts2 = np.float32([[y1+rndy[0],x1+rndy[0]],
                      [y2+rndy[1],x1+rndx[1]],
                      [y1+rndy[2],x2+rndx[2]]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(cols,rows))
    #print(dst.shape)
    dst = dst[:,:,np.newaxis]
    return dst
fig,axs = plt.subplots(1,2,figsize=(10,3))

axs[0].axis('off')
axs[0].imshow(test_img.squeeze(),cmap='gray')
axs[0].set_title('original')

axs[1].axis('off')
axs[1].imshow(test_dst.squeeze(),cmap='gray')
axs[1].set_title('warped')
print('shape in/out:',test_img.shape,test_dst.shape)
```

    shape in/out: (32, 32, 1) (32, 32, 1)



![png](https://img-blog.csdn.net/20180514103841163?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



```python
def random_brightness(img):
    shifted = img+1.0
    img_max_value = max(shifted.flatten())
    max_coef = 2.0/img_max_value
    min_coef = max_coef - 0.1
    coef = np.random.uniform(min_coef,max_coef)
    dst = shifted*coef - 1.0
    return dst

test_dst = random_brightness(test_img)
fig,axs = plt.subplots(1,2,figsize=(10,3))


axs[0].axis('off')
axs[0].imshow(test_img.squeeze(),cmap='gray')
axs[0].set_title('original')

axs[1].axis('off')
axs[1].imshow(test_dst.squeeze(),cmap='gray')
axs[1].set_title('warped')
print('shape in/out:',test_img.shape,test_dst.shape)
```

    shape in/out: (32, 32, 1) (32, 32, 1)



![png](https://img-blog.csdn.net/20180514103849639?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



```python
# histogram of label frequency (once again,before data augmentation)
hist,bins = np.histogram(y_train,bins=n_classes)
width = 0.7*(bins[1]-bins[0])
center = (bins[:-1]+bins[1:])/2
plt.bar(center,hist,align='center',width=width)
plt.show()
```


![png](https://img-blog.csdn.net/2018051410385926?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



```python
print(np.bincount(y_train))
print('minimum samples for any label:',min(np.bincount(y_train)))
```

    [ 180 1980 2010 1260 1770 1650  360 1290 1260 1320 1800 1170 1890 1920
      690  540  360  990 1080  180  300  270  330  450  240 1350  540  210
      480  240  390  690  210  599  360 1080  330  180 1860  270  300  210
      210]
    minimum samples for any label: 180



```python
print('X,y shapes:',X_train_normalized.shape,y_train.shape)
# input_indices map to output_indices
input_indices = []
output_indices = []

for class_n in range(n_classes):
    class_indices = np.where(y_train == class_n)
    n_samples = len(class_indices[0])
    if n_samples < 800:
        for i in range(800-n_samples):
            input_indices.append(class_indices[0][i%n_samples])
            output_indices.append(X_train_normalized.shape[0])
            new_img = X_train_normalized[class_indices[0][i%n_samples]]
            new_img = random_translate(random_scaling(random_warp(random_brightness(new_img))))
            X_train_normalized = np.concatenate((X_train_normalized,[new_img]),axis=0)
            y_train = np.concatenate((y_train,[class_n]),axis=0)
print('X,y shapes:',X_train_normalized.shape,y_train.shape)
```

    X,y shapes: (34799, 32, 32, 1) (34799,)
    X,y shapes: (46480, 32, 32, 1) (46480,)



```python
# histogram of label freguency
hist,bins = np.histogram(y_train,bins=n_classes)
width = 0.7*(bins[1]-bins[0])
center = (bins[:-1]+bins[1:])/2
plt.bar(center,hist,align='center',width=width)
plt.show()
```


![png](https://img-blog.csdn.net/2018051410390927?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



```python
# shuffle the training dataset
from sklearn.utils import shuffle

X_train_normalized,y_train = shuffle(X_train_normalized,y_train)
print('done')
```

    done



```python
# split validation dataset off from training dataset
from sklearn.model_selection import train_test_split
X_train,X_validation,y_train,y_validation = train_test_split(X_train_normalized,y_train,
                                                             test_size=0.20,random_state=42)
```


```python
import tensorflow as tf
EPOCHS = 60
BATCH_SIZE = 100
```

    /home/ora/anaconda3/envs/tensorflow/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters



```python
#我们首先实现超参数
from tensorflow.contrib.layers import flatten

def LeNet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1
    
    #Layer1: Concolutiona 1 input :32x32x1   output:28x28x6
    W1 = tf.Variable(tf.truncated_normal(shape=(5,5,1,6),mean=mu,stddev=sigma))
    x = tf.nn.con2d(x,W1,strides=[1,1,1,1],padding='VALID')
    b1 = tf.Variable(tf.zeros(6))
    x = tf.nn.bias_add(x,b1)
    print('layer 1 shape:',x.get_shape())
    
    #Activation
    x = tf.nn.relu(x)
    # Pooling input=28x28x6 Output = 14x14x6
    x = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    
    #Layer 2:Convolutional Output = 10x10x16
    W2 = tf.Variable(tf.truncated_normal(shape=(5,5,6,16),mean=mu,stddev=sigma))
    x = tf.nn.conv2d(x,W2,strides=[1,1,1,1],padding='VALID')
    b2 = tf.Variable(tf.zeros(16))
    x = tf.nn.bias_add(x,b2)
    #Activation
    x = tf.nn.relu(x)
    # Pooling input :10x10x16,output=5x5x16
    x = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    
    # Flatten Input:5x5x16 output=400
    x = flatten(x)
    # Layer 3: Fully Connected Input = 400 Output=120
    W3 = tf.Variable(tf.truncated_normal(shape=(400,120),mean=mu,stddev=sigma))
    b3 = tf.Variable(tf.zeros(120))
    x = tf.add(tf.matmul(x,W3),b3)  
    # Activation
    x = tf.nn.relu(x)
    #Dropout
    x = tf.nn.dropout(x,keep_prob)
    # Layer 4:Fully Connected Input=120 Output = 84
    W4 = tf.Variable(tf.truncated_normal(shape=(120,84),mean=mu,stddev = sigma))
    b4 = tf.Variable(tf.zeros(84))
    x = tf.add(tf.matmul(x,W4),b4)
    
    # Activation
    x = tf.nn.relu(x)
    # Dropout
    x = tf.nn.dropout(x,keep_prob)
    
    #Layer 5:Fully Connected Input =84 Output=43
    W5 = tf.Variable(tf.truncated_normal(shape=(84,43),mean=mu,stddev=sigma))
    b5 = tf.Variable(tf.zeros(43))
    logits = tf.add(tf.matmul(x,W5),b5)
    
    return logits
print('done')
```

    WARNING:tensorflow:From /home/ora/anaconda3/envs/tensorflow/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/base.py:198: retry (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use the retry module or similar alternatives.
    done



```python
def LeNet2(x):
    mu =0
    sigma = 0.1
    #Layer 1: Convolution 1 Input :32x32x1 Output:28x28x6
    W1 = tf.Variable(tf.truncated_normal(shape=(5,5,1,6),mean=mu,stddev=sigma),name='W1')
    x = tf.nn.conv2d(x,W1,strides=[1,1,1,1],padding='VALID')
    b1 = tf.Variable(tf.zeros(6),name='b1')
    x = tf.nn.bias_add(x,b1)
    print('layer 1 shape:',x.get_shape())
    
    #Activation
    x = tf.nn.relu(x)
    # Pooling Input:28x28x6 output:14x14x6
    x = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    layer1 = x
    
    #Layer2:Convolutiona 1 Output=10x10x16
    W2 = tf.Variable(tf.truncated_normal(shape=(5,5,6,16),mean=mu,stddev=sigma),name='W2')
    x = tf.nn.conv2d(x,W2,strides=[1,1,1,1],padding='VALID')
    b2 = tf.Variable(tf.zeros(16),name='b2')
    x = tf.nn.bias_add(x,b2)
    
    #Activation
    x = tf.nn.relu(x)
    #Pooling Input=10x10x16 Output=5x5x16
    x = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    layer2 = x
    
    #Layer3 Convolutional Output = 1x1x400
    W3 = tf.Variable(tf.truncated_normal(shape=(5,5,16,400),mean=mu,stddev=sigma),name='W3')
    x = tf.nn.conv2d(x,W3,strides=[1,1,1,1],padding='VALID')
    b3 = tf.Variable(tf.zeros(400),name='b3')
    x = tf.nn.bias_add(x,b3)
    
    #TODO:Activation
    x = tf.nn.relu(x)
    layers3 = x
    
    #TODO:Flateen Input:5x5x16 Output:400
    layer2flat = flatten(layer2)
    print('layer2flat shape:',layer2flat.get_shape())
    
    #Flatten x Input =1x1x400 Output = 400
    xflat = flatten(x)
    print('xflat shape:',xflat.get_shape())
    
    #Concat layer2flat and x Input=400+400 Output=800
    x = tf.concat([xflat,layer2flat],1)
    print('x shape:',x.get_shape())
    
    #Dropout
    x = tf.nn.dropout(x,keep_prob)
    
    #Layer4:Fully Connected Input:800,Output:43
    W4 = tf.Variable(tf.truncated_normal(shape=(800,43),mean=mu,stddev=sigma),name='W4')
    b4 = tf.Variable(tf.zeros(43),name='b4')
    logits = tf.add(tf.matmul(x,W4),b4)
    
    return logits
print('done')
```

    done



```python
tf.reset_default_graph()
x = tf.placeholder(tf.float32,(None,32,32,1))
y = tf.placeholder(tf.int32,(None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y,43)
print('done')
```

    done



```python
rate = 0.0009
logits = LeNet2(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)
```

    layer 1 shape: (?, 28, 28, 6)
    layer2flat shape: (?, 400)
    xflat shape: (?, 400)
    x shape: (?, 800)



```python
correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(one_hot_y,1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
saver = tf.train.Saver()
def evaluate(X_data,y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0,num_examples,BATCH_SIZE):
        batch_x,batch_y = X_data[offset:offset+BATCH_SIZE],y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation,feed_dict={x:batch_x,y:batch_y,keep_prob:1.0})
        total_accuracy+=(accuracy*len(batch_x))
    return total_accuracy/num_examples
print('done')
```

    done



```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    print('Training...')
    for i in range(EPOCHS):
        X_train,y_train = shuffle(X_train,y_train)
        for offset in range(0,num_examples,BATCH_SIZE):
            batch_x,batch_y = X_train[offset:offset+BATCH_SIZE],y_train[offset:offset+BATCH_SIZE]
            sess.run(training_operation,feed_dict={x:batch_x,y:batch_y,keep_prob:0.5})
        validation_accuracy = evaluate(X_validation,y_validation)
        print('EPOCH{}...'.format(i+1))
        print('Validation Accuracy={:.3f}'.format(validation_accuracy))
    saver.save(sess,'./lenet')
    print('Model saved')
```

    Training...
    EPOCH1...
    Validation Accuracy=0.870
    ......
    EPOCH58...
    Validation Accuracy=0.991
    EPOCH59...
    Validation Accuracy=0.992
    EPOCH60...
    Validation Accuracy=0.992
    Model saved



```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver2 = tf.train.import_meta_graph('./lenet.meta')
    saver2.restore(sess,'./lenet')
    test_accuracy = evaluate(X_test_normalized,y_test)
    print('Test Set Accuracy={:.3f}'.format(test_accuracy))
```

    INFO:tensorflow:Restoring parameters from ./lenet
    Test Set Accuracy=0.947


# Test a model on New Images



```python
import matplotlib.image as mpimg
import glob
import os

IMG_HEIGHT = 32
IMG_WIDTH = 32

def get_list_files(my_dir,f_ext):
    list_f = []
    for file in os.listdir(my_dir):
        if file.endswith('.'+f_ext):
            list_f.append(file)
    return list_f

my_dir = 'extra'

file_list = get_list_files(my_dir,'png')

X_extra = np.zeros((len(file_list),IMG_HEIGHT,IMG_WIDTH,3),dtype='uint8')
fig,axs = plt.subplots(5,2,figsize=(10,5))
fig.subplots_adjust(hspace=.2,wspace=.001)
axs = axs.ravel()
for idx,file in enumerate(file_list):
    img = cv2.imread(my_dir+'/'+file)
    img = cv2.resize(img,(32,32))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    axs[idx].imshow(img)
    X_extra[idx] = img
my_images = X_extra
my_images_gry = np.sum(my_images/3,axis=3,keepdims=True)
my_images_normalized = (my_images_gry-128)/128
print(my_images_normalized.shape)
```

    (32, 32, 3)
    (32, 32, 3)
    (32, 32, 3)
    (32, 32, 3)
    (32, 32, 3)
    (32, 32, 3)
    (32, 32, 3)
    (32, 32, 3)
    (32, 32, 3)
    (32, 32, 3)
    (10, 32, 32, 1)



![png](https://img-blog.csdn.net/20180514104001125?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



# 总结
本文只是给出了交通标志识别的一种baseline方案，该方法对背景复杂的交通标志识别的效果一般。因为如果要对有冗余背景的交通标志图片进行识别，最好能先去除冗余背景。

