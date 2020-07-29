from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D, UpSampling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys
import os
import numpy as np


class Gan():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_cols, self.img_rows, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)
        # 判别模型
        self.adversarial = self.build_adversarial()
        self.adversarial.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

        # 生成模型
        self.generator = self.build_generator()
        gen_input = Input(shape=(self.latent_dim,))
        img = self.generator(gen_input)

        self.adversarial.trainable = False
        validity = self.adversarial(img)
        self.combine = Model(gen_input, validity)
        self.combine.compile(optimizer=optimizer, loss="binary_crossentropy")

    def build_generator(self):
        """
        生成网络，用于生成手写字体图片（28，28，1）
        :return:
        """
        model = Sequential()
        # generator的输入  一串随机数字
        gen_input = Input(shape=(self.latent_dim,))
        # 全连接神经网络   参数100*256
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # 处理输出的shape (28, 28, 1)
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        img = model(gen_input)

        return Model(gen_input, img)

    def build_adversarial(self):
        """
        对抗网络，用于判断真伪
        :return:
        """
        model = Sequential()
        # 对抗网络的输入为generator的输出(28,28,1)
        ad_input = Input(shape=self.img_shape)
        # 将(28,28,1)平铺展开  784
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation="sigmoid"))

        validation = model(ad_input)
        return Model(ad_input, validation)

    def train(self, epochs, batch_size=128, sample_interval=50):
        # 加载minist数据  只要图片
        (x_train, _), (_, _) = mnist.load_data()

        # 数据处理
        # 此时的数据 28 * 28
        x_train = x_train / 127.5 - 1
        # 增加维度 28 * 28 * 1
        x_train = np.expand_dims(x_train, axis=3)
        # 标签 判别模型
        # 真实图片
        valid = np.ones((batch_size, 1))
        # generator 生成的假图
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            """
                随机选择batch_size个图片
                训练对抗网络
            """
            # 真图片
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]
            # 假图片 生成的图片
            gen_input = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(gen_input)

            # 损失函数
            ad_loss_real = self.adversarial.train_on_batch(imgs, valid)
            ad_loss_fake = self.adversarial.train_on_batch(gen_imgs, fake)
            ad_loss = 0.5 * np.add(ad_loss_fake, ad_loss_real)

            """
                训练generator
            """
            # 生成一串随机数字
            gen_input1 = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_loss = self.combine.train_on_batch(gen_input1, valid)

            # 打印损失
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, ad_loss[0], 100*ad_loss[1], gen_loss))

            # 每隔一段时间进行可视化
            if epoch % sample_interval == 0:
                self.sample_images(epoch)


    def sample_images(self, epoch):

        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()

if __name__ == '__main__':
    if not os.path.exists("./images"):
        os.makedirs("./images")
    gan = Gan()
    gan.train(epochs=30000, batch_size=256, sample_interval=200)