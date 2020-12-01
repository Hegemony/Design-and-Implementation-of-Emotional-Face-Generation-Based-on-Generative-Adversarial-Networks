# Design and Implementation of Emotional Face Generation Based on Generative Adversarial Networks

## Dependencies
- [Python3]
- [PyTorch]
- [Tensorflow] (optional for tensorboard)

#### Please refer to requirements.txt for specific installation package version.

## Datasets
- To download the CelebA dataset:
you can use this link: [https://pan.baidu.com/s/1dubdGfdZqRPS34-pVD1rog](https://pan.baidu.com/s/1dubdGfdZqRPS34-pVD1rog),
Extraction code:1q2w.

- To download the RaFD dataset: you must request access to the dataset from [the Radboud Faces Database website](http://www.socsci.ru.nl:8180/RaFD2/RaFD?p=main). 

## Pretrained Models

You can use this link: [https://pan.baidu.com/s/1EK9xRbzPmZwBXTlEvh6-pA](https://pan.baidu.com/s/1EK9xRbzPmZwBXTlEvh6-pA) to download pretrained models.
Extraction code:a1b1.

## Results

- 图片编辑

更改人脸属性，将输入图片依次转换为黑发，金发，棕发，男/女，年轻/衰老。

![img1](https://github.com/Hegemony/Design-and-Implementation-of-Emotional-Face-Generation-Based-on-Generative-Adversarial-Networks/blob/main/imgs_store/1.png)

- 离散表情生成

更改人脸表情，将输入图片依次转换为生气，轻蔑，厌恶，恐惧，快乐，中立，伤心，惊讶。

![img2](https://github.com/Hegemony/Design-and-Implementation-of-Emotional-Face-Generation-Based-on-Generative-Adversarial-Networks/blob/main/imgs_store/2.png)

- 连续表情生成

更改人脸表情：将输入图片按照目标域的人脸表情进行转换，转换的程度依次为20%， 40%， 60%， 80%， 100%。

![img3](https://github.com/Hegemony/Design-and-Implementation-of-Emotional-Face-Generation-Based-on-Generative-Adversarial-Networks/blob/main/imgs_store/3.png)
