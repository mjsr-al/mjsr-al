import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random as random
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dot, Multiply, Softmax, Reshape, Input, Flatten, Conv2DTranspose, Conv2D, GlobalAveragePooling2D,BatchNormalization, Lambda, MaxPooling2D, ReLU, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model, to_categorical

from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras import backend as K

import datetime
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
import tensorflow.keras as keras
import json

# Attractive, Blurry, Heavy_Makeup, Oval_Face, Pale_Skin, Young
# Big_Lips, Mouth_Slightly_Open, Smiling, Wearing_Lipstick

hyperparameters={
    'targets': ['image_id', 'Big_Lips', 'Mouth_Slightly_Open', 'Smiling', 'Wearing_Lipstick'],  ### MODIFY 
    'height': 224, 
    'width': 192 ,
    'channels': 3, 
    'batch_size': 16, 
    'epochs': 20, 
    'num_tasks': 4, ### MODIFY
    'initializer': 'he_uniform', 
    'reg_lambda': 1e-3, 
    'output': [2]*4, ### MODIFY === [2]*number of tasks
    'lr': 1e-5, ### MODIFY
    'is_trained': False, 
    'dropout_prob': 0.3,
    'enable_cs': False, 
    'enable_sluice': False,
    'initial_percent':0.1,
    'initial_train_epoch': 5, ### NEED TO SEE----- Done
    'increment_train_epoch':2, ### NEED TO SEE --- Done
    'uncertainity_repeat': 5, 
    'num_uncertain_elements': 5008, 
    'additional_epoch': 1, ### NEED TO SEE
    'pretraining_epochs': 2, ### -- make it 2 or originally 1 - Done
    'train_initial_batches': 8138, #6103, ### NEED TO SEE ------------- Done
    'enable_additional': False,
    'additional_attr_count':2,
    'all_updates':False,
    'initial_percent_val': 0.6 ### Added Now
    }

def preprocess(hyperparameters, attr, eval_partition):
    attr = attr[hyperparameters['targets']]
    attr = attr.replace(-1, 0)
    attr = attr.set_index('image_id')
    eval_partition = eval_partition.set_index('image_id')
    attr = attr.join(eval_partition)
    attr['image_id'] = attr.index
    
    for column in attr.columns[:-2]:
        k = to_categorical(attr[column])
        attr = attr.drop(column, axis=1)
        attr[column] = k.tolist()

    train = attr.loc[attr['partition']==0]
    val = attr.loc[attr['partition']==1]
    test = attr.loc[attr['partition']==2]
    
    train = train.drop('partition', axis=1)
    val = val.drop('partition', axis=1)
    test = test.drop('partition', axis=1)
    
    train = train[:(len(train)//hyperparameters['batch_size'])*hyperparameters['batch_size']]
    val = val[:(len(val)//hyperparameters['batch_size'])*hyperparameters['batch_size']]
    test = test[:(len(test)//hyperparameters['batch_size'])*hyperparameters['batch_size']]
    
    return (train, val, test)

def load_generator(df, shuffle=True):
    
    # image_id = '000014.jpg'
    image_path = '../input/celeba-dataset/img_align_celeba/img_align_celeba/'
    data_gen = ImageDataGenerator(rescale=1/255.0)
    
    generator = data_gen.flow_from_dataframe(dataframe = df, 
                                     directory=image_path, 
                                     x_col = 'image_id', 
                                     y_col=hyperparameters['targets'][1:], 
                                     class_mode = 'multi_output',
                                     target_size=(hyperparameters['height'], hyperparameters['width']), 
                                     batch_size = hyperparameters['batch_size'],shuffle=shuffle)
    
    return generator

def generate_generator_multiple(generator,dir1, dir2, df1,df2,batch_size, img_height,img_width,shuffle=True):
    # labelled 
    genX1 = generator.flow_from_dataframe(df1, dir1,
                                          x_col = 'image_id', 
                                          y_col=hyperparameters['targets'][1:], 
                                          class_mode = 'multi_output',
                                          target_size=(img_height,img_width), 
                                          batch_size = hyperparameters['batch_size'],
                                          shuffle=shuffle)
    # train --- make it train generator
    genX2 = generator.flow_from_dataframe(df2, dir2,
                                          x_col = 'image_id', 
                                          y_col=hyperparameters['targets'][1:], 
                                          class_mode = 'multi_output',
                                          target_size=(img_height, img_width), 
                                          batch_size = batch_size,
                                          shuffle=shuffle)
    while True:
            X1,y1 = genX1.next()
            X2,_ = genX2.next()
            yield X1, X2, y1  #Yield both images and their mutual label     

# changed input to 360+360
def discriminator():
    
    X = Input((720,), name='input_disc')
    
    x = Dense(units = 512, activation = None, name='dense_1_disc')(X)
    x = BatchNormalization(name = 'bn_1_disc')(x)
    x = ReLU(name='relu_1_disc')(x)
    x = Dense(units = 512, activation = None, name='dense_2_disc')(x)
    x = BatchNormalization(name = 'bn_2_disc')(x)
    x = ReLU(name='relu_2_disc')(x)
    x = Dense(units = 1, activation = 'sigmoid', name='output_disc')(x)
    
    return X,x

def addConvBlock(num_filters, kernel_size, hyperparameters, pool_size, tops, stride, pad, pool_stride, isPool , i):
    
    for task_id in range(hyperparameters['num_tasks']+hyperparameters['enable_additional']*hyperparameters['additional_attr_count']):
        
        tops[task_id] = Conv2D(num_filters, kernel_size=kernel_size, name = 'conv'+str(i)+'_'+str(task_id),  strides=(stride, stride), padding = pad, kernel_initializer=hyperparameters['initializer'], kernel_regularizer=regularizers.l1(hyperparameters['reg_lambda']))(tops[task_id])
        tops[task_id] = ReLU(name='relu'+str(i)+'_'+str(task_id))(tops[task_id])

        if (isPool==True):
            name = 'pool'+str(i)+'_'+str(task_id)
            tops[task_id] = MaxPooling2D(pool_size=(pool_size, pool_size), name = name, strides = (pool_stride, pool_stride), padding='valid')(tops[task_id])
            
    return tops


def combined_model():
    
    num_tasks = hyperparameters['num_tasks']+hyperparameters['enable_additional']*hyperparameters['additional_attr_count']
    
    x,y,z = hyperparameters['height'], hyperparameters['width'], hyperparameters['channels']
    X = Input((x,y,z), name = 'input_predictor_'+"_".join(str(num) for num in range(num_tasks)))
    tops = [X]*num_tasks

# ------------------------------------------- Block 1 BEGINS ------------------------------------

    tops = addConvBlock(40, 5, hyperparameters, 3, tops, 1, 'same', 2, True, 1)
    if hyperparameters["enable_cs"]:
        cs1 = CrossStitch(num_tasks,1, hyperparameters, True)(tops) 
        tops = tf.unstack(cs1, axis=0)
    
    tops = addConvBlock(60, 5, hyperparameters, 3, tops, 1, 'same', 2, True, 2)
    if hyperparameters["enable_cs"]:
        cs2 = CrossStitch(num_tasks, 2, hyperparameters, True)(tops)
        tops = tf.unstack(cs2, axis=0)
    
    tops = addConvBlock(80, 3, hyperparameters, 3, tops, 1, 'same', 2, True, 3)
    if hyperparameters["enable_cs"]:
        cs3 = CrossStitch(num_tasks, 3, hyperparameters, True)(tops)
        tops = tf.unstack(cs3, axis=0)
        
    
    tops = addConvBlock(100, 3, hyperparameters, 3, tops, 1, 'same', 2, True,  4)  
    if hyperparameters["enable_cs"]:
        cs4 = CrossStitch(num_tasks, 4, hyperparameters, True)(tops)
        tops = tf.unstack(cs4, axis=0)

# ------------------------------------------- Block 4 ENDS ------------------------------------
    
# ------------------------------------------- Block 5 BEGINS ------------------------------------
    tops = addConvBlock(140, 2, hyperparameters, 3, tops, 1, 'same', 2, True, 5)
    if hyperparameters["enable_cs"]:
        cs5 = CrossStitch(num_tasks, 5, hyperparameters, True)(tops)
        tops = tf.unstack(cs5, axis=0)

# ------------------------------------------- Block 5 ENDS ------------------------------------
    
    latents=[]
    weights=[]
    for task_id in range(num_tasks): 

        tops[task_id] = Flatten(name = 'flat'+'_'+str(task_id))(tops[task_id])

        tops[task_id] = Dense(units = 720, name = 'dense0'+'_'+str(task_id), kernel_initializer=hyperparameters['initializer'], kernel_regularizer=regularizers.l2(hyperparameters['reg_lambda']))(tops[task_id])
        if (task_id==0):
            tops[task_id] = ReLU(name='re_lu0'+'_'+str(0))(tops[task_id])
        else:
            tops[task_id] = ReLU(name='re_lu0'+'_'+str(task_id)+'_'+str(task_id))(tops[task_id])
            
            
        tops[task_id] = Dense(units = 360, name = 'dense1'+'_'+str(task_id), kernel_initializer=hyperparameters['initializer'], kernel_regularizer=regularizers.l2(hyperparameters['reg_lambda']))(tops[task_id])
        if (task_id==0):
            tops[task_id] = ReLU(name='re_lu'+'_'+str(0))(tops[task_id])
        else:
            tops[task_id] = ReLU(name='re_lu'+'_'+str(task_id)+'_'+str(task_id))(tops[task_id])
                                                        
        latents.append(tops[task_id])
        weights.append(tops[task_id])
        
        tops[task_id] = Dense(units = 180, name = 'dense2'+'_'+str(task_id), kernel_initializer=hyperparameters['initializer'], kernel_regularizer=regularizers.l2(hyperparameters['reg_lambda']))(tops[task_id])
        if (task_id==0):
            tops[task_id] = ReLU(name='re_lu2'+'_'+str(0))(tops[task_id])
        else:
            tops[task_id] = ReLU(name='re_lu2'+'_'+str(task_id)+'_'+str(task_id))(tops[task_id])
        

    # added joined weights
    joint = Concatenate(name='joined_1')(weights)
    joint = Dense(units = 360, name="joined_2",kernel_initializer=hyperparameters['initializer'], kernel_regularizer=regularizers.l2(hyperparameters['reg_lambda']))(joint)
    joint = Dense(units = num_tasks, name="joined_3",activation = 'softmax', kernel_initializer=hyperparameters['initializer'], kernel_regularizer=regularizers.l2(hyperparameters['reg_lambda']))(joint)
    tops = taskEmbeddings(5)([tops, joint])
       
    for task_id in range(num_tasks):
        tops[task_id] = Dense(units = 90, name = 'dense3'+'_'+str(task_id), kernel_initializer=hyperparameters['initializer'], kernel_regularizer=regularizers.l2(hyperparameters['reg_lambda']))(tops[task_id])
        if (task_id==0):
            tops[task_id] = ReLU(name='re_lu3'+'_'+str(0))(tops[task_id])
        else:
            tops[task_id] = ReLU(name='re_lu3'+'_'+str(task_id)+'_'+str(task_id))(tops[task_id])
    

    for task_id in range(num_tasks): 
        tops[task_id] = Dense(units = 2, name='output'+'_'+str(task_id),activation = 'softmax', kernel_initializer=hyperparameters['initializer'], kernel_regularizer=regularizers.l2(hyperparameters['reg_lambda']))(tops[task_id])    
        
    return X, tops, latents, joint

class CrossStitch(tf.keras.layers.Layer):

    def __init__(self, num_tasks,layer_num, hyperparameters, trainable, *args, **kwargs):
        self.num_tasks = num_tasks
        self.layer_num = layer_num
        self.hyperparameters = hyperparameters
        self.trainable = trainable
        super(CrossStitch, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = []
        for task_num in range(self.hyperparameters['num_tasks']):
            self.kernel.append(self.add_weight(name='cs_kernel_'+str(task_num)+'_'+str(self.layer_num), shape=(1, self.hyperparameters['num_tasks']),initializer=CrossStitchWeightInitializer(task_num, self.hyperparameters['num_tasks']),trainable=self.trainable))

        if self.hyperparameters['enable_additional'] == True:
            for task_num in range(self.hyperparameters['num_tasks'], self.num_tasks):
                self.kernel.append(self.add_weight(name='cs_kernel_'+str(task_num)+'_'+str(self.layer_num), shape=(1, self.num_tasks),initializer=CrossStitchWeightInitializer(task_num, self.num_tasks),trainable=self.trainable))
        super(CrossStitch, self).build(input_shape)  

    def call(self, input_feature_maps):        
        if (len(input_feature_maps)!=self.num_tasks):
            print("ERROR IN CROSS-STITCH")
      
        output_feature_maps = []
        for current_task in range(self.hyperparameters['num_tasks']):
            output = tf.math.scalar_mul(self.kernel[current_task][0,current_task], input_feature_maps[current_task])
            for other_task in range(self.hyperparameters['num_tasks']):
                if (current_task==other_task):
                    continue    
                output+= tf.math.scalar_mul(self.kernel[current_task][0,other_task], input_feature_maps[other_task])
            output_feature_maps.append(output)
        
        if self.hyperparameters['enable_additional'] == True:
            for current_task in range(self.hyperparameters['num_tasks'], self.num_tasks):
                output = tf.math.scalar_mul(self.kernel[current_task][0,current_task], input_feature_maps[current_task])
                for other_task in range(self.num_tasks):
                    if (current_task==other_task):
                        continue
                    output+= tf.math.scalar_mul(self.kernel[current_task][0,other_task], input_feature_maps[other_task])
                output_feature_maps.append(output)
        return tf.stack(output_feature_maps, axis=0)
  
    def compute_output_shape(self, input_shape):
        return [self.num_tasks] + input_shape

    def get_config(self):
        base_config = super(CrossStitch, self).get_config()
        base_config['num_tasks'] = self.num_tasks
        base_config['layer_num'] = self.task_num
        base_config['trainable'] = self.trainable
        base_config['hyperparameters'] = self.hyperparameters

        return base_config
    
class CrossStitchWeightInitializer(tf.keras.initializers.Initializer):

    def __init__(self, index, num_tasks):
        self.index=index
        self.num_tasks = num_tasks

    def __call__(self, shape, dtype=None):
        lis = [[0]*self.num_tasks]
        lis[0][self.index]=1
        return tf.Variable(lis, dtype=tf.float32)

    def get_config(self):  # To support serialization
        return {'index':self.index, 'num_tasks':self.num_tasks}
    
def variationalAutoEncoder(embed_size=360):
    # input
    X = Input((hyperparameters['height'],hyperparameters['width'],hyperparameters['channels']), name="input_vae")
    
    # encoder
    x = Conv2D(128, kernel_size=(4,4), strides = (2,2), padding='same', use_bias=False, name='conv_1_vae')(X)
    x = BatchNormalization(name='bn_1_vae')(x)
    x = ReLU(name='relu_1_vae')(x)
    x = Conv2D(256, kernel_size=(4,4), strides = (2,2), padding='same', use_bias=False, name='conv_2_vae')(x)
    x = BatchNormalization(name='bn_2_vae')(x)
    x = ReLU(name='relu_2_vae')(x)
    x = Conv2D(512, kernel_size=(4,4), strides = (2,2), padding='same', use_bias=False, name='conv_3_vae')(x)
    x = BatchNormalization(name='bn_3_vae')(x)
    x = ReLU(name='relu_3_vae')(x)
    x = Conv2D(1024, kernel_size=(4,4), strides = (2,2), padding='same', use_bias=False, name='conv_4_vae')(x)
    x = BatchNormalization(name='bn_4_vae')(x)
    x = ReLU(name='relu_4_vae')(x)
    
    x = Flatten(name='flatten_1_vae')(x)
    
    mu = Dense(units = embed_size, activation = None, name='dense_1_vae')(x)
    
    #logvar = Dense(units = 32, activation = None)(x)
    # check what to be done here --->  ??
    #stds = Lambda(lambda x: x * 0.5)(logvar)
    #stds = tf.keras.backend.exp(stds)
    #epsilon = tf.keras.backend.random_normal((32,))
    #m = tf.keras.layers.Multiply()([stds,epsilon])
    #latents = tf.keras.layers.Add()([m,mu])
        
    # decoder
    z = Dense(units = 14*12*1024, activation = None, name='dense_2_vae')(mu)
    z = Reshape((14,12,1024), name='reshape_vae')(z)
    z = Conv2DTranspose(512, kernel_size=(4,4), strides = (2,2), padding='same', use_bias=False, name='conv_5_vae')(z)
    z = BatchNormalization(name='bn_5_vae')(z)
    z = ReLU(name='relu_5_vae')(z)
    z = Conv2DTranspose(256, kernel_size=(4,4), strides = (2,2), padding='same', use_bias=False, name='conv_6_vae')(z)
    z = BatchNormalization(name='bn_6_vae')(z)
    z = ReLU(name='relu_6_vae')(z)
    z = Conv2DTranspose(128, kernel_size=(4,4), strides = (2,2), padding='same', use_bias=False, name='conv_7_vae')(z)
    z = BatchNormalization(name='bn_7_vae')(z)
    z = ReLU(name='relu_7_vae')(z)
    z = Conv2DTranspose(3, strides=(2,2), kernel_size=(1,1), name='conv_8_vae')(z)
    
    return X, z, mu

def predictor():
    X, tops, latents,joint = combined_model()
    model = Model(inputs=X, outputs=[tops,latents,joint])
    return model

class taskEmbeddings(tf.keras.layers.Layer):
    def __init__(self, num_layers, **kwargs):
        self.num_layers = num_layers
        super(taskEmbeddings, self).__init__(**kwargs)

    def build(self, input_shape):
        super(taskEmbeddings, self).build(input_shape)

    def call(self, x):

        dynamic_weights = x[1] # 16x5
        output_feature_maps = x[0] # 16x5x90

        data = np.identity(len(output_feature_maps), dtype = np.float32)
        #onehotencoder = OneHotEncoder() # categorical_features = [0]
        #data = onehotencoder.fit_transform(np.arange(len(output_feature_maps)).reshape((-1,1))).toarray()
        weighted_output=[]
        for output in range(len(output_feature_maps)):
            one_hot = tf.constant(data[output])
            one_hot = tf.keras.backend.reshape(one_hot, shape=(-1, len(output_feature_maps)))
            #one_hot = tf.repeat(one_hot, repeats=[hyperparameters['batch_size']], axis = 0)
            product1 = Dot(axes=1)([dynamic_weights, one_hot]) # 16x1
            product1 = tf.repeat(product1, repeats=180, axis = 1)
            product2 = Multiply()([product1, output_feature_maps[output]])
            weighted_output.append(product2)

        return weighted_output

    #def compute_output_shape(self, input_shape):
    #    return input_shape

    def get_config(self):
        base_config = super(taskEmbeddings, self).get_config()
        base_config['num_layers'] = self.num_layers
        return base_config
    
class ActiveLearning(keras.Model):
    def __init__(self, discriminator, generator, predictor, trackers, alpha):
        super(ActiveLearning, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.predictor = predictor
        self.trackers = trackers
        self.alpha=alpha
        self.prev_loss = 10000

    def compile(self, d_optimizer, g_optimizer, p_optimizer):
        super(ActiveLearning, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.p_optimizer = p_optimizer
                
    def train_step(self, real_images):
        
        # get labelled_x, unlabelled_x and labelled_y
        
        x = real_images
        labelled_x = x[0]
        unlabelled_x = x[1]
        labelled_y = x[2]
        
        ##### TRAIN THE PREDICTOR #####
        
        
        # Compute output and latents
        with tf.GradientTape() as tape:
            labelled_prediction_y, _, _ = self.predictor(labelled_x, training=True)
            predictor_loss = keras.losses.categorical_crossentropy(labelled_y, labelled_prediction_y) # ----> 1
        
        # Compute gradients
        trainable_vars = self.predictor.trainable_variables
        gradients = tape.gradient(predictor_loss, trainable_vars)

        # Update weights
        self.p_optimizer.apply_gradients(zip(gradients, trainable_vars)) 
        
        # ------------------------------------------------------------------------------------------------
        
        ##### TRAIN THE GENERATOR #####
        
        # Create labels for VAE
        labelled_disc_true = np.ones((hyperparameters['batch_size'],1))
        unlabelled_disc_fake = np.ones((hyperparameters['batch_size'],1))
        
        # Compute VAE outputs
        with tf.GradientTape() as tape:
            # Compute generator o/p
            labelled_vae_y, labelled_vae_latent = self.generator(labelled_x)
            unlabelled_vae_y, unlabelled_vae_latent = self.generator(unlabelled_x)
            
            # Calculate loss for VAE
            labelled_vae_loss = keras.losses.mean_squared_error(labelled_x, labelled_vae_y) # ----> 2
            unlabelled_vae_loss = keras.losses.mean_squared_error(unlabelled_x, unlabelled_vae_y) # ----> 2
            
            # Compute predictor o/p
           # _, labelled_predictor_latent = self.predictor(labelled_x)
            #_, unlabelled_predictor_latent = self.predictor(unlabelled_x)
            
            # Average out the latents for 5 tasks --- SHOULD I?
            # labelled_vae_latent = math_ops.mean(ops.convert_to_tensor(labelled_vae_latent), axis=0)
            # unlabelled_vae_latent = math_ops.mean(ops.convert_to_tensor(unlabelled_vae_latent), axis=0)
            #labelled_predictor_latent = math_ops.mean(ops.convert_to_tensor(labelled_predictor_latent), axis=0)
            #unlabelled_predictor_latent = math_ops.mean(ops.convert_to_tensor(unlabelled_predictor_latent), axis=0)
            
            # Join vae and predictor latents
            #labelled_disc_in = tf.concat([labelled_vae_latent,labelled_predictor_latent],axis=1)
            #unlabelled_disc_in = tf.concat([unlabelled_vae_latent,unlabelled_predictor_latent],axis=1)            
            
            # Compute disc o/p
            #labelled_disc_y = self.discriminator(labelled_disc_in)
            #unlabelled_disc_y = self.discriminator(unlabelled_disc_in)
            
            # Calculate loss for disc
            #labelled_disc_loss = keras.losses.binary_crossentropy(labelled_disc_true, labelled_disc_y) # ----> 3
            #unlabelled_dic_loss = keras.losses.binary_crossentropy(unlabelled_disc_fake, unlabelled_disc_y) # ----> 3
            
            # Compute total VAE loss
            #disc_loss = labelled_disc_loss + unlabelled_dic_loss
            vae_loss = labelled_vae_loss + unlabelled_vae_loss #+ (self.advisory_param*disc_loss)
        
        # Compute gradients
        trainable_vars = self.generator.trainable_variables
        gradients = tape.gradient(vae_loss, trainable_vars)
        
        # Update weights
        self.g_optimizer.apply_gradients(zip(gradients, trainable_vars))         
        
        # ------------------------------------------------------------------------------------------------
        
        ##### TRAIN THE DISCRIMINATOR #####
        
        # Create disc labels
        labelled_disc_true = np.ones((hyperparameters['batch_size'],1))
        unlabelled_disc_true = np.zeros((hyperparameters['batch_size'],1))
        
        # Compute VAE latents
        _, labelled_vae_latent = self.generator(labelled_x, training = False)
        _, unlabelled_vae_latent = self.generator(unlabelled_x, training = False)
        
        # Compute predictor latents
        _, labelled_predictor_latent, _ = self.predictor(labelled_x, training=False)
        _, unlabelled_predictor_latent, _ = self.predictor(unlabelled_x, training=False)
        
        # Average out the latents for 5 tasks --- SHOULD I?
        # labelled_vae_latent = math_ops.mean(ops.convert_to_tensor(labelled_vae_latent), axis=0)
        # unlabelled_vae_latent = math_ops.mean(ops.convert_to_tensor(unlabelled_vae_latent), axis=0)
        labelled_predictor_latent = math_ops.mean(ops.convert_to_tensor(labelled_predictor_latent), axis=0)
        unlabelled_predictor_latent = math_ops.mean(ops.convert_to_tensor(unlabelled_predictor_latent), axis=0)
        
        # Join vae and predictor latents
        labelled_disc_in = tf.concat([labelled_vae_latent,labelled_predictor_latent],axis=1)
        unlabelled_disc_in = tf.concat([unlabelled_vae_latent,unlabelled_predictor_latent],axis=1)
        
        # Compute disc output
        with tf.GradientTape() as tape:
            labelled_disc_y = self.discriminator(labelled_disc_in,training=True)
            unlabelled_disc_y = self.discriminator(unlabelled_disc_in,training=True)
            
            labelled_disc_loss = keras.losses.binary_crossentropy(labelled_disc_true, labelled_disc_y) # ----> 3
            unlabelled_dic_loss = keras.losses.binary_crossentropy(unlabelled_disc_true, unlabelled_disc_y) # ----> 3
            
            disc_loss = labelled_disc_loss + unlabelled_dic_loss
        
        # Compute gradients
        trainable_vars = self.discriminator.trainable_variables
        gradients = tape.gradient(disc_loss, trainable_vars)
        
        # Update weights
        self.d_optimizer.apply_gradients(zip(gradients, trainable_vars)) 
    
        # ------------------------------------------------------------------------------------------------
        
        # Computing Metrics
        
        # For predictor
        
        self.trackers['loss_tracker_predictor'].update_state(labelled_y, labelled_prediction_y)
        self.trackers['acc_metric_predictor'].update_state(labelled_y, labelled_prediction_y)
        
        #loss_tracker_predictor.update_state(labelled_y, labelled_prediction_y)
        #acc_metric_predictor.update_state(labelled_y, labelled_prediction_y)
        
        for i in range(hyperparameters['num_tasks']):
            self.trackers['individual_loss_tracker_predictor'][i].update_state(labelled_y[i], labelled_prediction_y[i])
            self.trackers['individual_acc_metric_predictor'][i].update_state(labelled_y[i], labelled_prediction_y[i])
                
        # For VAE
        self.trackers['loss_tracker_generator'].update_state(labelled_x, labelled_vae_y)
        self.trackers['loss_tracker_generator'].update_state(unlabelled_x, unlabelled_vae_y)
        # For Discriminator
        self.trackers['loss_tracker_disc'].update_state(labelled_disc_true,labelled_disc_y)
        self.trackers['loss_tracker_disc'].update_state(unlabelled_disc_true,unlabelled_disc_y)
        self.trackers['acc_tracker_disc'].update_state(labelled_disc_true,labelled_disc_y)
        self.trackers['acc_tracker_disc'].update_state(unlabelled_disc_true,unlabelled_disc_y)


            
        ret_dic = {"loss_predictor_total": self.trackers['loss_tracker_predictor'].result(), # loss_tracker_predictor.result(), 
                   "acc_predictor":self.trackers['acc_metric_predictor'].result(), # acc_metric_predictor.result(), 
                   "loss_VAE":  self.trackers['loss_tracker_generator'].result(), # loss_tracker_generator.result(),
                   "loss_disc": self.trackers['loss_tracker_disc'].result(), # loss_tracker_disc.result(),
                   "acc_disc": self.trackers['acc_tracker_disc'].result()} # acc_tracker_disc.result()}
        
        for i in range(hyperparameters['num_tasks']):
            ret_dic["loss_predictor_"+str(i)] = self.trackers['individual_loss_tracker_predictor'][i].result() # individual_loss_tracker_predictor[i].result()
        for i in range(hyperparameters['num_tasks']):
            ret_dic["acc_predictor_"+str(i)] = self.trackers['individual_acc_metric_predictor'][i].result() # individual_acc_metric_predictor[i].result()
            
        return ret_dic
    
    def call(self, x):
        return
    
    def test_step(self, real_images):
        
        x = real_images
        labelled_x = x[0]
        labelled_y = x[1]
        
        # Predictor step
        labelled_prediction_y, labelled_predictor_latent, _ = self.predictor(labelled_x, training=False)
        
        # Generator step
        labelled_vae_y, labelled_vae_latent = self.generator(labelled_x, training=False)
        
        # Discriminator step
        labelled_predictor_latent = math_ops.mean(ops.convert_to_tensor(labelled_predictor_latent), axis=0)
        labelled_disc_in = tf.concat([labelled_vae_latent,labelled_predictor_latent],axis=1)
        
        labelled_disc_y = self.discriminator(labelled_disc_in,training=False)
        
        # Updating metrics
        # For Predictor
        self.trackers['loss_tracker_predictor'].update_state(labelled_y, labelled_prediction_y)
        self.trackers['acc_metric_predictor'].update_state(labelled_y, labelled_prediction_y)
        
        for i in range(hyperparameters['num_tasks']):
            self.trackers['individual_loss_tracker_predictor'][i].update_state(labelled_y[i], labelled_prediction_y[i])
            self.trackers['individual_acc_metric_predictor'][i].update_state(labelled_y[i], labelled_prediction_y[i])
            
        self.trackers['loss_tracker_generator'].update_state(labelled_x, labelled_vae_y)
        
        
        # For Discriminator
        labelled_disc_true = np.ones((hyperparameters['batch_size'],1))
        self.trackers['loss_tracker_disc'].update_state(labelled_disc_true,labelled_disc_y)
        self.trackers['acc_tracker_disc'].update_state(labelled_disc_true,labelled_disc_y)
        
        
        ret_dic = {"loss_predictor_total": self.trackers['loss_tracker_predictor'].result(), # loss_tracker_predictor.result(), 
                   "acc_predictor": self.trackers['acc_metric_predictor'].result(), # acc_metric_predictor.result(), 
                   "loss_VAE":  self.trackers['loss_tracker_generator'].result(), # loss_tracker_generator.result(),
                   "loss_disc": self.trackers['loss_tracker_disc'].result(), # loss_tracker_disc.result(),
                   "acc_disc": self.trackers['acc_tracker_disc'].result()} # acc_tracker_disc.result()}
        
        for i in range(hyperparameters['num_tasks']):
            ret_dic["loss_predictor_"+str(i)] = self.trackers['individual_loss_tracker_predictor'][i].result() # individual_loss_tracker_predictor[i].result()
        for i in range(hyperparameters['num_tasks']):
            ret_dic["acc_predictor_"+str(i)] = self.trackers['individual_acc_metric_predictor'][i].result() # individual_acc_metric_predictor[i].result()
                    
#         ret_dic = {"loss_predictor_total": loss_tracker_predictor.result(), 
#                    "acc_predictor": acc_metric_predictor.result(), 
#                    "loss_VAE": loss_tracker_generator.result(),
#                    "loss_disc": loss_tracker_disc.result(),
#                    "acc_disc": acc_tracker_disc.result()}
        
#         for i in range(hyperparameters['num_tasks']):
#             ret_dic["loss_predictor_"+str(i)] = individual_loss_tracker_predictor[i].result()
#         for i in range(hyperparameters['num_tasks']):
#             ret_dic["acc_predictor_"+str(i)] = individual_acc_metric_predictor[i].result()
            
        return ret_dic        
    
    def predict_step(self, real_images):
        unlabelled_x, unlabelled_y = real_images
        
        # Predictor step
        unlabelled_prediction_y, unlabelled_predictor_latent, joint_weights = self.predictor(unlabelled_x, training=False)
        
        # Generator step
        unlabelled_vae_y, unlabelled_vae_latent = self.generator(unlabelled_x, training=False)
        
        # Discriminator step
        unlabelled_predictor_latent = math_ops.mean(ops.convert_to_tensor(unlabelled_predictor_latent), axis=0)
        unlabelled_disc_in = tf.concat([unlabelled_vae_latent,unlabelled_predictor_latent],axis=1)
        
        unlabelled_disc_y = self.discriminator(unlabelled_disc_in,training=False)
        
        return unlabelled_prediction_y, unlabelled_disc_y, unlabelled_y, joint_weights

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.trackers["loss_tracker_predictor"], self.trackers["acc_metric_predictor"], self.trackers["loss_tracker_generator"], self.trackers["loss_tracker_disc"], self.trackers["acc_tracker_disc"]] + self.trackers["individual_loss_tracker_predictor"] + self.trackers["individual_acc_metric_predictor"]
    
def uncertainity(probs, weights):
    lis = []
    lis_output = []
    for i in range(hyperparameters['num_tasks']):
        attr_output = probs[i]
        w = weights[:,i]
        k = -1* np.sum(attr_output*np.log(attr_output),axis=1)
        lis_output.append(k)
        lis.append(w*k)
    
    variance = np.var(np.array(lis),axis=0)
    return np.array(lis).sum(axis=0), variance

def getIndices(output, hyperparameters ,pretrain=False):
    if pretrain == True:
        count =  hyperparameters['train_initial_batches']*hyperparameters['batch_size']
        if ((output<=0.5).sum())>=count:
            sort = np.argwhere(output<=0.5)[:,0]
            return sort
        else:
            selection = (int((hyperparameters['train_initial_batches']*hyperparameters['batch_size'])/1000)+1)*1000
            sort = np.argpartition((output)[:,0], selection)
            return sort[:selection]
    else:
        count = hyperparameters['num_uncertain_elements']
        if ((output<=0.5).sum())>=count:
            sort = np.argwhere(output<=0.5)[:,0]
            return sort
        else:
            selection = (int(hyperparameters['num_uncertain_elements']/1000)+1)*1000
            sort = np.argpartition((output)[:,0], selection)
            return sort[:selection]
        
def divide_data(train, initial = False):
    num_samples = train.values.shape[0]
    
    if initial:
        idx = random.sample(list(np.arange(num_samples)), ((int(hyperparameters['initial_percent_val']*num_samples)//hyperparameters['batch_size'])*hyperparameters['batch_size']))
    else:
        idx = random.sample(list(np.arange(num_samples)), ((int(hyperparameters['initial_percent']*num_samples)//hyperparameters['batch_size'])*hyperparameters['batch_size']))

    return pd.DataFrame(train.values[idx,:], columns=train.columns), idx

attr = pd.read_csv('../input/celeba-dataset/list_attr_celeba.csv')
eval_partition = pd.read_csv('../input/celeba-dataset/list_eval_partition.csv')

break_point_ep = {'3': 5e-4,'6': 5e-4,'10': 1e-5}
splits = [0.1,0.15,0.2,0.25,0.3,0.35,0.40]

# defining metrics

trackers = {
    "loss_tracker_predictor": tf.keras.metrics.CategoricalCrossentropy(name="loss_predictor_total"),
    "acc_metric_predictor": tf.keras.metrics.CategoricalAccuracy(name="acc_predictor"),
    "individual_loss_tracker_predictor": [tf.keras.metrics.CategoricalCrossentropy(name="loss_predictor_"+str(i)) for i in range(hyperparameters['num_tasks'])],
    "individual_acc_metric_predictor": [tf.keras.metrics.CategoricalAccuracy(name="acc_predictor_"+str(i)) for i in range(hyperparameters['num_tasks'])],
    "loss_tracker_generator": tf.keras.metrics.MeanSquaredError(name='loss_VAE'),
    "loss_tracker_disc":  tf.keras.metrics.BinaryCrossentropy(name='loss_disc'),
    "acc_tracker_disc": tf.keras.metrics.BinaryAccuracy("acc_disc")
}

class CalculatingPredictions(tf.keras.callbacks.Callback):
    def __init__(self, preds, test_gen, train_gen, lr, is_validation=False):
        self.preds = preds
        self.train_gen = train_gen
        self.test_gen = test_gen
        self.lr = lr
        self.is_validation=is_validation

    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.train_gen, verbose=1)  
        pre = p[0]
        out = p[2]
        h = keras.losses.mae(out, pre).numpy().mean()
        print(h)

        if (h>=self.model.prev_loss):
            self.model.alpha = self.model.alpha + (self.lr*h)
        else:
            self.model.alpha = self.model.alpha - (self.lr*h)
        self.model.prev_loss = h
        variance_weight=np.array([self.model.prev_loss,self.model.alpha])
        if (self.is_validation == True):
            np.save("./saved_history/val_params" +str(epoch)+ '.npy', variance_weight)
        else:
            np.save("./saved_history/params" + str(epoch) +'.npy', variance_weight)

        print(self.model.alpha)
        predict=self.model.evaluate(self.test_gen)
        print(predict)
        self.preds.append(predict)
        k = np.array(self.preds)
        if (self.is_validation==True):
            np.save("./saved_history/AL_model_from_scratch_non_cs_validation"+ str(epoch)+ ".npy", k)
        else:
            np.save("./saved_history/AL_model_from_scratch_non_cs_"+ str(epoch)+ ".npy", k)
        date = datetime.datetime.now().strftime("%d - %b - %y - %H:%M:%S")
        
        if (self.is_validation==False and epoch%2==0):
            self.model.predictor.save_weights("./saved_history/models/pred_model_from_scratch_vanilla_epoch"+str(epoch)+'.h5')
            self.model.discriminator.save_weights("./saved_history/models/disc_model_from_scratch_vanilla_epoch"+str(epoch)+'.h5')
            self.model.generator.save_weights("./saved_history/models/vae_model_from_scratch_vanilla_epoch"+str(epoch)+'.h5')

if not os.path.isdir('logs'):
    os.makedirs('logs')
if not os.path.isdir('saved_history/models'):
    os.makedirs('saved_history/models')
if not os.path.isdir('results'):
    os.makedirs('results')


def startTraining(trackers, splits, break_point_ep,  validation_first = False, load_pred_model = False, load_vae = False, load_disc = False, further_training=False):
    preds=[] ####
    validation_train_history=[] # new
    date = datetime.datetime.now().strftime("%d - %b - %y - %H:%M:%S")

    filename = "AL_model_from_scratch_non_cs_"+date ###
    logdir = "logs/" + filename

    filepath = "./saved_history/model/AL_model_from_scratch_non_cs"
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False, mode='max', period = 1)
    csv_logger = CSVLogger('./saved_history/training_results_AL_model_from_scratch_non_cs.csv', separator = ',', append=True)
    pre_train_logger = CSVLogger('./saved_history/pre_training_results_AL_model_from_scratch_non_cs.csv', separator = ',', append=True)
    tensorboard_callback = TensorBoard(log_dir = logdir)

    pre_tensorboard_callback = TensorBoard(log_dir =" logs/pre_"+filename)

    # Instantiate components
    # defining my predictor
    pred_model = predictor()
    pred_model.compile(optimizer = keras.optimizers.SGD(learning_rate=hyperparameters['lr'],
                                                        clipnorm=1.0 ))
    if load_pred_model: #####
        print("Predictor weights loading ...")
        pred_model.load_weights("./saved_history/models/pred_model_from_scratch_vanilla_4.h5", by_name=True)
    
    # defining my discriminator
    disc_in, disc_out = discriminator()
    disc = Model(inputs = disc_in, outputs = disc_out)
    disc.compile(optimizer = keras.optimizers.SGD(learning_rate=hyperparameters['lr'],
                                                  clipnorm=1.0 ))
    if load_disc: ###
        print("Discriminator weights loading ...")
        disc.load_weights("./saved_history/models/disc_model_from_scratch_vanilla_4.h5", by_name=True)
        
    # defining my generator
    X, z, mu = variationalAutoEncoder()
    vae = Model(inputs = X, outputs = [z,mu])
    vae.compile(optimizer = keras.optimizers.RMSprop(learning_rate=hyperparameters['lr'],
                                                     clipnorm=1.0 ))
    if load_vae: ####
        print("VAE weights loading ...")
        vae.load_weights("./saved_history/models/vae_model_from_scratch_vanilla_4.h5" ,by_name = True)
    
    # Instantiate AL model
    AL_model = ActiveLearning(discriminator=disc, generator=vae, predictor=pred_model, trackers = trackers, alpha=1)
    AL_model.compile(
        d_optimizer=keras.optimizers.SGD(learning_rate=hyperparameters['lr'],clipnorm=1.0 ),
        g_optimizer= keras.optimizers.RMSprop(learning_rate=hyperparameters['lr'],clipnorm=1.0),
        p_optimizer=keras.optimizers.SGD(learning_rate=hyperparameters['lr'],clipnorm=1.0 ))

    print('model loaded')

    train_imggen = ImageDataGenerator(rescale = 1./255)
    image_path = '../input/celeba-dataset/img_align_celeba/img_align_celeba/'
    train, val, test = preprocess(hyperparameters, attr, eval_partition)
    train_gen_full = load_generator(train, False)
    val_gen = load_generator(val)
    test_gen = load_generator(test, False)

    if validation_first == True:
        logger = CSVLogger('./saved_history/pretraining_results_AL_model_from_scratch_non_cs.csv', separator = ',', append=True)
        pre_tensorboard_callback = TensorBoard(log_dir =" logs/pre_"+filename)
        
        labelled_pretrain, idx_prelabelled = divide_data(val, initial=True)
        idx_preunlabelled = list(np.setdiff1d(list(range(val.shape[0])), idx_prelabelled))
        unlabelled_pretrain = pd.DataFrame(val.values[idx_preunlabelled,:], columns=val.columns)
       
        pretrain_gen = generate_generator_multiple(generator=train_imggen,
                                               dir1=image_path,
                                               dir2=image_path,
                                               df1 = labelled_pretrain,
                                               df2 = unlabelled_pretrain,
                                               batch_size=hyperparameters['batch_size'],
                                               img_height=hyperparameters['height'],
                                               img_width=hyperparameters['width'])
        
        labelled_pretrain_gen  = load_generator(labelled_pretrain, False)
        
        val_history = AL_model.fit(pretrain_gen, epochs = hyperparameters['pretraining_epochs'], steps_per_epoch = 744, callbacks = [CalculatingPredictions(preds, test_gen, labelled_pretrain_gen, 0.01, True) , pre_tensorboard_callback, logger], verbose=1)
        validation_train_history.append(val_history.history)
        with open("./saved_history/pretraining_history_list.json", 'w') as f:
            json.dump(validation_train_history, f, indent=2)
            
        labelled_train, idx_labelled = divide_data(train)
        idx_unlabelled = list(np.setdiff1d(list(range(train.shape[0])), idx_labelled))
        unlabelled_train = pd.DataFrame(train.values[idx_unlabelled,:], columns=train.columns)        
        train_gen = generate_generator_multiple(generator=train_imggen,
                                               dir1=image_path,
                                               dir2=image_path,
                                               df1 = labelled_train,
                                               df2 = unlabelled_train, 
                                               batch_size=hyperparameters['batch_size'],
                                               img_height=hyperparameters['height'],
                                               img_width=hyperparameters['width'])
        unlabelled_gen = load_generator(unlabelled_train, False)    
        labelled_train_gen = load_generator(labelled_train, False)
        
    elif further_training==True:

        idx_labelled = np.load("./saved_history/idx_labelled_4.npy")
        labelled_train = train.iloc[idx_labelled, :]
        idx_unlabelled = list(np.setdiff1d(list(range(train.shape[0])), idx_labelled))
        unlabelled_train = pd.DataFrame(train.values[idx_unlabelled,:], columns=train.columns)
        train_gen = generate_generator_multiple(generator=train_imggen,
                                               dir1=image_path,
                                               dir2=image_path,
                                               df1 = labelled_train,
                                               df2 = unlabelled_train,
                                               batch_size=hyperparameters['batch_size'],
                                               img_height=hyperparameters['height'],
                                               img_width=hyperparameters['width'])
        unlabelled_gen = load_generator(unlabelled_train, False)
        labelled_train_gen = load_generator(labelled_train, False)
    else:
        labelled_train, idx_labelled = divide_data(train)
        idx_unlabelled = list(np.setdiff1d(list(range(train.shape[0])), idx_labelled))
        unlabelled_train = pd.DataFrame(train.values[idx_unlabelled,:], columns=train.columns)        
        train_gen = generate_generator_multiple(generator=train_imggen,
                                               dir1=image_path,
                                               dir2=image_path,
                                               df1 = labelled_train,
                                               df2 = unlabelled_train, 
                                               batch_size=hyperparameters['batch_size'],
                                               img_height=hyperparameters['height'],
                                               img_width=hyperparameters['width'])
        unlabelled_gen = load_generator(unlabelled_train, False)    
        labelled_train_gen = load_generator(labelled_train, False)

    history_list=[]

    if further_training==True:
        num_batches = 10 #idx_labelled.shape[0]//hyperparameters['batch_size']
        tensorboard_callback = TensorBoard(log_dir = 'AL_model_from_scratch_non_cs_09 - Jan - 21 - 12:50:36')

        iteration =5
        epoch_num = 13

        history = AL_model.fit(train_gen,initial_epoch = epoch_num, epochs=epoch_num+7, steps_per_epoch = num_batches, validation_data=val_gen,callbacks = [CalculatingPredictions(preds, test_gen,labelled_train_gen, 0.01), csv_logger], verbose = 1)
        history_list.append(history.history)
        pred_model.save_weights("./saved_history/models/pred_model_from_scratch_vanilla_"+str(iteration)+'.h5')
        disc.save_weights("./saved_history/models/disc_model_from_scratch_vanilla_"+str(iteration)+'.h5')
        vae.save_weights("./saved_history/models/vae_model_from_scratch_vanilla_"+str(iteration)+'.h5')
        with open("./saved_history/history_list_only_"+str(iteration)+".json", 'w') as f:
            json.dump(history_list, f, indent=2)

        with open("./saved_history/preds_only_"+str(iteration)+".json", 'w') as f:
            json.dump(preds, f, indent=2)
    else:
        test_predictions=[]
        indices_list = []
        epoch_num = 0
        num_batches= int(train.shape[0] * (hyperparameters['initial_percent']) / hyperparameters['batch_size'])

        for iteration in range(len(splits)):
            print(iteration)

            if iteration==0:
                # Initial training ---- change
                history = AL_model.fit(train_gen, epochs=hyperparameters['initial_train_epoch'], steps_per_epoch = num_batches, validation_data=val_gen,callbacks = [CalculatingPredictions(preds, test_gen, labelled_train_gen, 0.01), csv_logger, tensorboard_callback], verbose = 1)
                history_list.append(history.history)
                epoch_num+=hyperparameters['initial_train_epoch']
            else:
                # Increment training --- change
                history = AL_model.fit(train_gen, initial_epoch = epoch_num, epochs=epoch_num+hyperparameters['increment_train_epoch'], steps_per_epoch = num_batches, validation_data=val_gen,callbacks = [CalculatingPredictions(preds, test_gen, labelled_train_gen, 0.01), csv_logger, tensorboard_callback], verbose = 1)
                history_list.append(history.history)
                epoch_num+=hyperparameters['increment_train_epoch']

            pred_model.save_weights("./saved_history/models/pred_model_from_scratch_vanilla_"+str(iteration)+'.h5')
            disc.save_weights("./saved_history/models/disc_model_from_scratch_vanilla_"+str(iteration)+'.h5')
            vae.save_weights("./saved_history/models/vae_model_from_scratch_vanilla_"+str(iteration)+'.h5')

            # append indices
            indices_list.append(idx_labelled)
            inc = int(train.shape[0]*0.05 /  hyperparameters['batch_size'])
            num_batches+= inc
            print('Number of batches added:' , inc)

            with open("./saved_history/history_list_"+str(iteration)+".json", 'w') as f:
                json.dump(history_list, f, indent=2)

            with open("./saved_history/preds_"+str(iteration)+".json", 'w') as f:
                json.dump(preds, f, indent=2)

            np.save("./saved_history/idx_labelled_"+str(iteration)+".npy",np.array(idx_labelled))

            if (iteration!=(len(splits)-1)): # last iteration
                # print("checking for uncertainities")
                # Get uncertainities

                # _, disc_output,_ ,_= AL_model.predict(unlabelled_gen, verbose=1)

                # unlabelled_indices = getIndices(disc_output, hyperparameters) # 0-49999
                # outputs = []
                # weights = []
                # for i in range(hyperparameters['uncertainity_repeat']):
                #    print("Done "+str(i))
                #    p = AL_model.predict(unlabelled_gen, verbose=1)
                #    predictor_output= p[0]
                #    joint_weights = p[3]
                #    weights.append(np.array(joint_weights)[unlabelled_indices])
                #    unlabelled_predictions = np.array(predictor_output)[:,unlabelled_indices,:]
                #    outputs.append(unlabelled_predictions)

                # avg_output = np.mean(np.array(outputs), axis=0)
                # avg_weights = np.mean(np.array(weights), axis=0)    

                # t = uncertainity(avg_output, avg_weights)
                # t=t[0] + (t[1] * AL_model.alpha)
                # uncertain_indices = np.argpartition(t, -1*hyperparameters['num_uncertain_elements'])[-1*hyperparameters['num_uncertain_elements']:]
                # img_indice = np.array(unlabelled_indices)[uncertain_indices]

                # k = np.array(idx_unlabelled)[img_indice]
                
                k = random.sample(idx_unlabelled, hyperparameters['num_uncertain_elements'])
                
                
                idx_labelled = idx_labelled+list(k)
                labelled_train = pd.DataFrame(train.values[idx_labelled,:], columns=train.columns)
                idx_unlabelled = list(np.setdiff1d(idx_unlabelled, k))
                unlabelled_train = pd.DataFrame(train.values[idx_unlabelled,:], columns=train.columns)
                train_gen = generate_generator_multiple(generator=train_imggen,
                                                               dir1=image_path,
                                                               dir2=image_path,
                                                               df1 = labelled_train,
                                                               df2 = unlabelled_train,
                                                               batch_size=hyperparameters['batch_size'],
                                                               img_height=hyperparameters['height'],
                                                               img_width=hyperparameters['width'])
                unlabelled_gen = load_generator(unlabelled_train, False)
                labelled_train_gen = load_generator(labelled_train, False)
    
    return history_list, pred_model, vae, disc, AL_model, indices_list, preds

history_list, pred_model, vae, disc, Al_model, indices_list, preds = startTraining(trackers, splits, break_point_ep, True, False, False, False, False)
