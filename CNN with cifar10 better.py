import gc
gc.enable()

####################
# 避免占满
####################

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus) # 打印gpu列表
print()

for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

####################
# 加载数据
####################

from tensorflow.keras.datasets import cifar10

# 归一化数据
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 独热编码
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

####################
# 构造模型
####################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Flatten, Dense, Dropout

def get_model(): # vgg16
    
    model = Sequential()
    
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model

####################
# 获取模型
####################

model = get_model()

from tensorflow.keras.optimizers import Adam

model.compile( optimizer = Adam( learning_rate = 1e-4 ),
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'] )

model.summary()

####################
# 图表
####################

import matplotlib.pyplot as plt
plt.rcParams['savefig.dpi'] = 300 # 图片像素

class PlotProgress(tf.keras.callbacks.Callback):

    def __init__(self, entity = ['loss', 'accuracy']):
        self.entity = entity

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        
        self.losses = []
        self.val_losses = []

        self.accs = []
        self.val_accs = []

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        # 损失函数
        self.losses.append(logs.get('{}'.format(self.entity[0])))
        self.val_losses.append(logs.get('val_{}'.format(self.entity[0])))
        # 准确率
        self.accs.append(logs.get('{}'.format(self.entity[1])))
        self.val_accs.append(logs.get('val_{}'.format(self.entity[1])))

        self.i += 1
        
        plt.figure( figsize = (6, 3) )

        plt.subplot(121)
        plt.plot(self.x, self.losses, label="{}".format(self.entity[0]))
        plt.plot(self.x, self.val_losses, label="val_{}".format(self.entity[0]))
        plt.legend()
        plt.grid()
        plt.title('loss')

        plt.subplot(122)
        plt.plot(self.x, self.accs, label="{}".format(self.entity[1]))
        plt.plot(self.x, self.val_accs, label="val_{}".format(self.entity[1]))
        plt.legend()
        plt.grid()
        plt.title('accuracy')

        plt.tight_layout() # 减少白边
        plt.savefig('visualization.jpg')
        plt.close() # 关闭

####################
# 回调函数
####################

# 绘图函数
plot_progress = PlotProgress(entity=['loss', 'accuracy'])

# 早退
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True)

####################
# 训练
####################

model.fit( x_train, y_train,
           validation_data = (x_test, y_test),

           epochs = 10000, batch_size = 32,

           callbacks = [plot_progress, early_stopping],

           verbose = 2, # 2 一次训练就显示一行
           shuffle = True, # 再次打乱

           # max_queue_size = 1000,
           workers = 4, # 多进程核心数
           use_multiprocessing = True, # 多进程
           )


