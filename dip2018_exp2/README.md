# dip2018_exp2

## requirements

python>=3.5

pytorch==0.3.1

## pretrained model

```
import torchvision.models as models
alexnet = models.alexnet(pretrained=True)
```

## baseline

1. 直接finetune（只调最后一层）

2. 在原来的fc1000上加一层fc50（只调最后一层）

3. 提出training set的feature之后，validation的label根据training set feature中nearest neighbour决定

4. data argumentation

## ideas

1. with baseline (2), init weight with causal inference

2. 
