---
layout: post
title: "MLP 모델에 대한 CPU vs GPU 수행 시간 비교"
tags: [cpu vs gpu, deep learning, machine learning, mlp train time]
image:
  path: /images/abstract-1.jpg
  feature: abstract-1.jpg
comments: true
---

Neural Network 모델을 무언가 해보겠다면 GPU가 필수적입니다. 학습 수행 속도를 비약적으로 절감하기 때문에 우리는 좋은 성능을 가진 GPU를 필수적으로 구비해서 Deep Learning으로 무언가를 하려고 시도합니다.  GPU가 CPU 보다 학습을 얼마나 빨리 끝내는지를 MLP(Multi Layer Perceptron) 모델을 간단하게 구현해서 CPU vs GPU 학습 수행 시간을 비교하겠습니다.

제가 가지고 있는 GPU는 **GerForce GTX 660** 모델입니다. Deep Learning 모델을 수행하기에는 제가 가진 데스크톱의 H/W는 굉장히 구형 모델임을 감안하고 비교 결과를 확인하시기 바랍니다. **MLP 모델을 케라스**로 구현하고  **데이터셋은 mnist** 를 사용했습니다.

## CPU vs GPU 수행 시간 비교

아래 결과에서 중요한 점은 항상 GPU가 학습 수행 시간이 짧지 않다는 것입니다. **학습 파라미터 수가 134,794 이하까지는 오히려 CPU에서 학습에 걸리는 시간이 더 짧습니다**. 약 3백만 개의 학습 파라미터에서는 **GPU가 CPU 보다 5배 수행 시간이** 단축되었습니다. 학습 파라미터 수가 적으면 HOST DRAM에서 GPU DRAM으로 복사하는 오버헤드 때문에 오히려 성능이 더 저하되는 것 같습니다. 결론은 자신의 **모델이 작고 가볍다면 CPU를 통해서 학습해도 충분**하다는 것입니다.



| **전체 학습 파라미터 수** | **CPU (i5 750)** |                       | **GPU (GeForce GTX 660)** |                       |
| ------------------------- | ---------------- | --------------------- | ------------------------- | --------------------- |
|                           | **accuracy**     | **train time (secs)** | **accuracy**              | **train time (secs)** |
| 3,230                     | 12.0523          | 0.6008                | 32.7011                   | 0.5369                |
| 6,514                     | 13.4265          | 0.8016                | 32.8508                   | 0.7783                |
| 13,274                    | 15.1539          | 0.9227                | 32.9818                   | 0.9208                |
| 27,562                    | 15.8237          | 0.9579                | 32.7519                   | 0.9589                |
| 59,210                    | 21.3826          | 0.9725                | 33.0307                   | 0.9736                |
| 134,794                   | 31.7006          | 0.9782                | 32.4855                   | 0.9777                |
| 335,114                   | 59.6029          | 0.9813                | 33.4356                   | 0.9816                |
| 932,362                   | 128.9178         | 0.9797                | 37.803                    | 0.982                 |
| 2,913,290                 | 354.0091         | 0.981                 | 60.7474                   | 0.9807                |

## 시스템 환경

### Hardware

| 항목            | 내용                               |
| :-------------- | :--------------------------------- |
| CPU             | Intel i5 CPU 750 2.67GHz (4스레드) |
| GPU             | GeForce GTX 660                    |
| RAM             | 4GB                                |
| Primary Storage | WD Green SSD 240GB                 |

### Software

| 항목           | 내용                        |
| :------------- | :-------------------------- |
| OS             | Ubuntu 16.04.5              |
| Kernel         | 4.15.0-38                   |
| tensorflow     | 1.10.0 (local source build) |
| tensorflow-gpu | 1.10.0 (local source build) |
| keras          | 2.2.4                       |
| cuda           | 9.0                         |
| cudnn          | 7.3                         |

### Dataset

| 항목                  | 내용                                        |
| :-------------------- | :------------------------------------------ |
| mnist (손글씨 이미지) | train samples : 60000, test samples : 10000 |

## Keras로 구현한 MLP 모델 코드

```python
import time
import random
import keras
import argparse

from keras.datasets import mnist
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn import preprocessing

random.seed(777)

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--units', type=int, default=1024, required=False)
args = parser.parse_args()

# hyper parameters
batch_size = 128
epochs = 10
dense_units = args.units
dropout_rate = 0.2
optimizer = 'adam'

classes = 10

mnist_width = 28
mnist_height = 28
mnist_size = mnist_width * mnist_height

# load mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('x_train shape :', x_train.shape)
print('y_train shape :', y_train.shape)
print('x_test shape :', x_test.shape)
print('y_test shape :', y_test.shape)

x_train = x_train.reshape(x_train.shape[0], mnist_size)
x_test = x_test.reshape(x_test.shape[0], mnist_size)

# apply min max scaling
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# to one-hot vector
y_train = keras.utils.to_categorical(y_train, classes)
y_test = keras.utils.to_categorical(y_test, classes)

# build model
model = Sequential()
model.add(Dense(dense_units, input_shape=(mnist_size,), activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(dense_units, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(dense_units, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(classes, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

start_time = time.time()
history = model.fit(x_train, y_train, batch_size=batch_size,
                    epochs=epochs, validation_split=0.1)
elapsed_time = time.time() - start_time

loss, accuracy = model.evaluate(x_test, y_test)

print('total params : {}, epochs : {}, train time : {:.4f} secs, evaluate accuracy : {:.4f}'.format(
    model.count_params(), epochs, elapsed_time, accuracy))
```

