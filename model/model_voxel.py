import tensorflow as tf
from tensorflow.keras import layers

def build_generator():
    model = tf.keras.Sequential()
    # 输入层：假设输入图片大小为 64x64x3
    model.add(layers.Conv2D(64, (5, 5), padding='same', input_shape=(64, 64, 3)))
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(128, (5, 5)))
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D())

    model.add(layers.Flatten())
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU())

    # 假设生成的点云有 1024 个点，每个点3个坐标
    model.add(layers.Dense(1024*3))
    model.add(layers.Reshape((1024, 3)))  # 输出层：将线性层的输出重塑为点云格式

    return model

generator = build_generator()
generator.summary()