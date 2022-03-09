# 论文分享：Planning in Stochastic Environments with a Learned Model (ICLR 2022 Spotlight) 

## 一、动机

MuZero在多个领域的控制任务上取得了SOTA的表现，然而

- Previous ~~model-based methods~~ *MuZero* is limited to a deterministic class of dynamic models.

- Many environments are inherently stochastic and may be poorly approximated by a deterministic model.

因此，本文将 *MuZero* 拓展为*Stochastic* *MuZero*；原文链接：https://openreview.net/forum?id=X6D9bAHhBQ1

## 二、MuZero

### 2.1 Model Training

![image-20220309101129037](https://raw.githubusercontent.com/tjuHaoXiaotian/Markdown4Zhihu/master/Data/P1-Stochastic_MuZero/image-20220309101129037.png)

### 2.2 Search

![image-20220309101237861](https://raw.githubusercontent.com/tjuHaoXiaotian/Markdown4Zhihu/master/Data/P1-Stochastic_MuZero/image-20220309101237861.png)

## 三、*Stochastic* *MuZero*

### 3.1 Stochastic Transition Model

![image-20220309101605259](https://raw.githubusercontent.com/tjuHaoXiaotian/Markdown4Zhihu/master/Data/P1-Stochastic_MuZero/image-20220309101605259.png)

![image-20220309101810091](https://raw.githubusercontent.com/tjuHaoXiaotian/Markdown4Zhihu/master/Data/P1-Stochastic_MuZero/image-20220309101810091.png)

![image-20220309101955955](https://raw.githubusercontent.com/tjuHaoXiaotian/Markdown4Zhihu/master/Data/P1-Stochastic_MuZero/image-20220309101955955.png)

### 3.2  Stochastic Search

![image-20220309102526741](https://raw.githubusercontent.com/tjuHaoXiaotian/Markdown4Zhihu/master/Data/P1-Stochastic_MuZero/image-20220309102526741.png)

## 4. 实验

![image-20220309102640682](https://raw.githubusercontent.com/tjuHaoXiaotian/Markdown4Zhihu/master/Data/P1-Stochastic_MuZero/image-20220309102640682.png)

![image-20220309102713354](https://raw.githubusercontent.com/tjuHaoXiaotian/Markdown4Zhihu/master/Data/P1-Stochastic_MuZero/image-20220309102713354.png)

![image-20220309102735629](https://raw.githubusercontent.com/tjuHaoXiaotian/Markdown4Zhihu/master/Data/P1-Stochastic_MuZero/image-20220309102735629.png)

