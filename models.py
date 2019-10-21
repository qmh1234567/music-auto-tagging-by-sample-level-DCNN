""" 
@ author: Qmh
@ file_name: models.py
@ time: 2019:09:23:08:38
""" 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAvgPool1D,Dense,Conv1D,BatchNormalization,Activation
from tensorflow.keras.layers import Reshape,MaxPool1D,Multiply,Input,GlobalMaxPool1D,Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Add,Flatten
import tensorflow as tf
import constants as c

WEIGHT_DECAY = 0.
AMPLIFYING_RATIO = 0.125
NUM_BLOCKS = 9

def squeeze_exciation(x,amplifying_ratio,name):
    num_features = x.shape[-1].value
    x = GlobalAvgPool1D(name=f'squeeze_{name}')(x)
    x = Reshape((1,num_features),name=f'reshape_{name}')(x)
    x = Dense(int(num_features*amplifying_ratio),activation='relu',name=f'ex0_{name}')(x)
    x = Dense(num_features,activation='sigmoid',name=f'ex1_{name}')(x)
    return x

def basic_block(x,num_features,name):
    x = Conv1D(num_features,kernel_size=3,padding='same',use_bias=True,
    kernel_regularizer=l2(WEIGHT_DECAY),name=f'conv_{name}')(x)
    x = BatchNormalization(name=f'norm_{name}')(x)
    x = Activation('relu',name=f'relu_{name}')(x)
    x = MaxPool1D(pool_size=3,name=f'pool_{name}')(x)
    return x

def se_block(x,num_features,name):
    x = basic_block(x,num_features,name)
    x = Multiply(name=f'scale_{name}')([x,squeeze_exciation(x,AMPLIFYING_RATIO,name)])
    return x

def Sample_CNN(input_shape,num_class):
    x_in = Input(input_shape,name='input')
    num_features = 128
    x = Conv1D(num_features,kernel_size=3,strides=3,padding='same',kernel_regularizer=l2(WEIGHT_DECAY),name='conv0')(x_in)
    x = BatchNormalization(name='norm0')(x)
    x = Activation('relu',name='relu0')(x)
    # stack se blocks
    for i in range(NUM_BLOCKS):
        num_features *= 2 if (i==2 or i==(NUM_BLOCKS-1)) else 1
        x = se_block(x,num_features,name=f'block_{i+1}')
    x = GlobalMaxPool1D(name='final_pool')(x)
    
    # the final two FCs
    x = Dense(x.shape[-1].value,name='fc1')(x)
    x = BatchNormalization(name='norm1')(x)
    x = Activation('relu',name='relu1')(x)
    x = Dropout(0.5,name='drop1')(x)
    x = Dense(num_class,activation='sigmoid',name='output')(x)

    return Model(inputs=x_in,outputs=x,name='sampleCNN')
    


def wavenetBlock(num_features,atrous_filter_size,atrous_rate):
    def f(input_):
        residual = input_
        x_tanh = Conv1D(num_features,atrous_filter_size,
            dilation_rate=atrous_rate,padding='valid',activation='tanh')(input_)
   
        x_sigmoid =Conv1D(num_features,atrous_filter_size,
           dilation_rate=atrous_rate,padding='valid',activation='tanh')(input_)

        merged = Multiply()([x_tanh,x_sigmoid])
        skip_out = Conv1D(num_features,1,activation='relu')(merged)
        out = Add()([skip_out,residual])
        return out,skip_out
    return f




# # 2 epochs auc = 0.856 prauc=30.7
# def basic_block1(x,num_features,size,rate,name):
#     # x = Conv1D(num_features,kernel_size=3,padding='same',use_bias=True,
#     # kernel_regularizer=l2(WEIGHT_DECAY),name=f'conv_{name}')(x)

#     x_tanh = Conv1D(num_features,size,
#             dilation_rate=rate,padding='valid',activation='tanh',name=f'conv_{name}_tanh')(x)
   
#     x_sigmoid =Conv1D(num_features,size,
#         dilation_rate=rate,padding='valid',activation='sigmoid',name=f'conv_{name}_sigmoid')(x)

#     merged = Multiply()([x_tanh,x_sigmoid])

#     x = Conv1D(num_features,1,activation='relu')(merged)

#     # x = BatchNormalization(name=f'norm_{name}')(x)
#     # x = Activation('relu',name=f'relu_{name}')(x)
#     # x = MaxPool1D(pool_size=3,name=f'pool_{name}')(x)
#     return x

# def proposed_CNN(input_shape,num_class):
#     x_in = Input(input_shape,name='input')
#     num_features = 128
#     # x = Conv1D(num_features,kernel_size=3,strides=3,padding='same',kernel_regularizer=l2(WEIGHT_DECAY),name='conv0')(x_in)
#     # x = BatchNormalization(name='norm0')(x)
#     # x = Activation('relu',name='relu0')(x)
#     # stack se blocks
#     x = x_in
#     k = 0
#     rate = 13
#     NUM_BLOCKS = 7
#     for i in range(NUM_BLOCKS):
#         x = Conv1D(num_features,kernel_size=3,strides=3,padding='same',kernel_regularizer=l2(WEIGHT_DECAY),name=f'conv0_{k}')(x)
#         x = BatchNormalization(name=f'norm_{k}')(x)
#         x = Activation('relu',name=f'relu0_{k}')(x)
#         # for r in range(4):
#         num_features *= 2 if (i==2 or i==(NUM_BLOCKS-1)) else 1
#         # num_features = 128
#         # print("r=",r)
#         x = basic_block1(x,num_features,3,rate,name=f'block_{k+1}')
#         # print(x.shape)
#         k += 1

#     x = GlobalMaxPool1D(name='final_pool')(x)
    
#     # the final two FCs
#     x = Dense(x.shape[-1].value,name='fc1')(x)
#     x = BatchNormalization(name='norm1')(x)
#     x = Activation('relu',name='relu1')(x)
#     x = Dropout(0.5,name='drop1')(x)
#     x = Dense(num_class,activation='sigmoid',name='output')(x)

#     return Model(inputs=x_in,outputs=x,name='sampleCNN')


# 添加残差
# def basic_block1(x,num_features,size,rate,name):
#     # 残差
#     original_x = x

#     num_features1 = num_features//4

#     x_tanh = Conv1D(num_features1,size,
#             dilation_rate=rate,padding='same',
#             kernel_regularizer=l2(WEIGHT_DECAY),name=f'conv_{name}_tanh')(x)
            
#     x_tanh = BatchNormalization()(x_tanh)
#     x_tanh = Activation('tanh')(x_tanh)

#     x_sigmoid =Conv1D(num_features1,size,
#         dilation_rate=rate,padding='same',
#         kernel_regularizer=l2(WEIGHT_DECAY),name=f'conv_{name}_sigmoid')(x)
#     x_sigmoid = BatchNormalization()(x_sigmoid)
#     x_sigmoid = Activation('sigmoid')(x_sigmoid)

#     merged = Multiply()([x_tanh,x_sigmoid])
    
#     x = Conv1D(num_features,1,padding='same')(merged)
#     x = BatchNormalization()(x)

#     x = Add()([original_x,x])
#     x = Activation('relu')(x)
#     return x


# no SE block auc=0.91
def residual_block(x,num_features,size,dilation_rate,name):
    # 残差
    original_x = x

    num_features1 = num_features//4

    merged_list = []

    # for rate in dilation_rate:
    rate = dilation_rate
    x = Conv1D(num_features1,size,
            dilation_rate=rate,padding='same',
            kernel_regularizer=l2(WEIGHT_DECAY),name=f'conv_{name}_{rate}')(x)  # 注意计算dilation_rate
        
    x = BatchNormalization()(x)

    x_tanh = Activation('tanh')(x)

    x_sigmoid = Activation('sigmoid')(x)

    merged = Multiply()([x_tanh,x_sigmoid])
    
    x = Conv1D(num_features,1,padding='same')(merged)
    x = BatchNormalization()(x)

    x = Add()([original_x,x])
    x = Activation('relu')(x)
    return x

# SE block
def SE_residual_block(x,num_features,size,dilation_rate,name):
    # 残差
    original_x = x

    num_features1 = num_features//4

    merged_list = []

    rate = dilation_rate
    x = Conv1D(num_features1,size,
            dilation_rate=rate,padding='same',
            kernel_regularizer=l2(WEIGHT_DECAY),name=f'conv_{name}_{rate}')(x)
            
    x = BatchNormalization()(x)

    x_tanh = Activation('tanh')(x)

    x_sigmoid = Activation('sigmoid')(x)

    merged = Multiply()([x_tanh,x_sigmoid])
    
    x = Conv1D(num_features,1,padding='same')(merged)
    x = BatchNormalization()(x)

    x = Multiply(name=f'scale_{name}')([x,squeeze_exciation(x,AMPLIFYING_RATIO,name)])

    x = Add()([original_x,x])
    x = Activation('relu')(x)
    return x



def Strided_Conv(x,num_features,k):
    x = Conv1D(num_features,kernel_size=3,strides=3,padding='same',kernel_regularizer=l2(WEIGHT_DECAY),name=f'conv0_{k}')(x)
    x = BatchNormalization(name=f'norm_{k}')(x)
    x = Activation('relu',name=f'relu0_{k}')(x)
    return x


def proposed_CNN(input_shape,num_class):
    x = x_in = Input(input_shape,name='input')
    num_features = 128
    # stack se blocks
    k = 1
    dilation_rate = 4 # dilation rate
    kernel_size = 3
    NUM_BLOCKS = 7
    start = 3  # (729,256)
    end = 7   # (27,256)
    for i in range(NUM_BLOCKS):
        num_features *= 2 if (i==start) else 1
        if i>= start and i < end:
        # 用不同的r
            x = Strided_Conv(x,num_features,k)
            x = residual_block(x,num_features,kernel_size,dilation_rate,name=f'block_{i+1}_{k+1}')
            # x = SE_residual_block(x,num_features,kernel_size,dilation_rate,name=f'block_{i+1}_{k+1}')
        else:
            x = Strided_Conv(x,num_features,k)
        k += 1

    x = GlobalMaxPool1D(name='final_pool')(x)
    
    # the final two FCs
    x = Dense(x.shape[-1].value,name='fc1')(x)
    x = BatchNormalization(name='norm1')(x)
    x = Activation('relu',name='relu1')(x)
    x = Dropout(0.5,name='drop1')(x)
    x = Dense(num_class,activation='sigmoid',name='output')(x)

    return Model(inputs=x_in,outputs=x,name='sampleCNN')

if __name__ == "__main__":
    input_shape = (59049,1)
    num_class = len(c.TAGS)
    model = Sample_CNN(input_shape,num_class)
    # model = proposed_CNN(input_shape,num_class)
    print(model.summary())

    
