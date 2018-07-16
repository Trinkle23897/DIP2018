# dip2018_exp2

## feature的处理

```
import numpy as np
d = np.load('src/feature.npy').item()
print(d)
# 会看到 feature 和 label
feature = d['feature']
label = d['label']
# TODO
```

## requirements

python: 兼容>=2.7和>=3.5 （请注意//和/的用法和print的用法，print最好用format）

pytorch==0.4.0

```
sudo pip install torch torchvision
```

## 运行

```
cd src
python main.py -d ../data --prefix try 
```

运行完之后会在当前目录下生成`try.pth`的模型文件，先存着。

## 文件结构

src: 代码，其中 model 的一些写法可参考 main.py 中的 baseline()

data: 数据，分为 train和 val，由于 val 尚未公布因此只有train，目前只能交叉验证了

material: 一些阅读材料，包括 Few-shot Learning 现在的 state-of-the-art

**一些训好的模型不要加到git上！**

## pretrained model

```
import torchvision.models as models
alexnet = models.alexnet(pretrained=True)
```

## baseline

1. 直接finetune（只调最后一层）done

2. 在原来的fc1000上加一层fc50（只调最后一层）

3. 提出training set的feature之后，validation的label根据training set feature中nearest neighbour决定

4. data argumentation

## ideas

1. with baseline (2), init weight with causal inference

2. 
