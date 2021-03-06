# Design and Implementation of Emotional Face Generation Based on Generative Adversarial Networks

## Dependencies
* [Python 3.7+](https://www.continuum.io/downloads)
* [PyTorch 1.1.0+](http://pytorch.org)
* [TensorFlow 1.14.0+](https://tensorflow.google.cn/) (optional for tensorboard)

#### Please refer to requirements.txt for specific installation package version.

## Datasets
- To download the CelebA dataset:
you can use this link: [https://pan.baidu.com/s/1dubdGfdZqRPS34-pVD1rog](https://pan.baidu.com/s/1dubdGfdZqRPS34-pVD1rog),
Extraction code:1q2w.

- To download the RaFD dataset: you must request access to the dataset from [the Radboud Faces Database website](http://www.socsci.ru.nl:8180/RaFD2/RaFD?p=main). 

## Use Own Datasets
- **Crop Face**：Extract face from datasets and crop face from images.

- **Obtain AUs Vector**: Use [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) to extract Action Units vectors from the cropped face. The extracted AUs is as follows:
> AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU07_r, AU09_r, AU10_r, AU12_r, AU14_r, AU15_r, AU17_r, AU20_r, AU23_r, AU25_r, AU26_r, AU45_r

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
