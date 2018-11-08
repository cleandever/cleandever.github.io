---
layout: post
title: "텐서보드 이벤트 파일에서 이미지 추출"
tags: [텐서보드, tensorboard, keras, tensorflow, 이미지저장, 이미지추출]
image:
  path: /images/abstract-2.jpg
comments: true
---

텐서플로우를 통해서 크고 복잡한 모델 사용하는 경우 이를 이해하고 디버깅하고 최적화하기 위해서 텐서보드 툴을 사용합니다. 텐서보드 툴을 통해서 학습 과정을 visualization 할 수 있습니다. 텐서보드는 웹 브라우저를 통해서 학습 과정 및 기록을 확인 할 수 있습니다.

![tensorboard](/images/2018/11/08_tensorboard/tensorboard_capture.png)

케라스에서 텐서보드를 연동하는 방법은 TensorBoard 클래스를 생성하고, fit() 호출시 콜백 함수로 지정을 해주면 됩니다.

```python
tb_hist = keras.callbacks.TensorBoard(log_dir='./tb_graph',
                                      histogram_freq=0,
                                      write_graph=True,
                                      write_images=True)

model.fit(x_train, y_train, batch_size=batch_size,
          epochs=epochs, validation_split=0.1, callbacks=[tb_hist])
```



학습이 시작되면 ./tb_graph 디렉토리 밑에 파일명이 "event.~" 시작하는 파일이 생성됩니다.

아래와 같이 실행해서 결과를 웹 브라우져를 통해서 확인할 수 있습니다.

```bash
tensorboard --logdir=[tensorboard event 파일]
```



텐서보드가 생성한 이벤트 파일에서 바로 loss or acc or val_loss or val_acc 등을 바로 이미지로 저장할 수 있습니다.



## 텐서보드가 생성한 이벤트 파일에서 바로 이미지 저장 코드

```python
import os
import argparse
import tensorflow as tf
import pylab


def save_image(event_filename, tag):
    # check file
    assert(os.path.exists(event_filename))

    values = list()
    for e in tf.train.summary_iterator(event_filename):
        for v in e.summary.value:
            if v.tag == tag:
                values.append(v.simple_value)

    pylab.clf()
    pylab.plot(range(len(values)), values)
    pylab.xlabel("epoch")
    pylab.ylabel(tag)

    image_filename = '{}/{}.png'.format(os.path.dirname(event_filename),
                                        tag)
    pylab.savefig(image_filename)
    print('saved image : ', image_filename)


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', required=True)
args = parser.parse_args()

save_image(args.filename, "loss")
save_image(args.filename, "acc")
save_image(args.filename, "val_loss")
save_image(args.filename, "val_acc")
```
