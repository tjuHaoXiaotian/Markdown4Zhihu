# MARL-5: [干货] 基于值分解的多智能体强化算法回顾



[TOC]



## 1. 一张大图

![一张大图](./banner.png)

此图按时间线涵盖了主要的基于值分解的多智体强化学习算法（multi-agent value function factorization methods），在本文中，首先回顾在CTDE（Centralized Training with Decentralized Execution ）范式下，满足IGM（Individual-Global-Max）条件的4个代表算法：VDN [1], QMIX [2], QTRAN [3] 和 QPLEX [7]，其他文章后续再做介绍。



## 2. 核心矛盾：Q函数 “拟合能力强弱” 与 “最值求解难易程度”

### Optimal Bellman Equation

本篇主要涉及的5个算法属于Q-learning based算法，本质在于利用动态规划求解“最优贝尔曼方程”（optimal bellman equation），集中式（Centralized Q-learning）视角下，最优贝尔曼方程形式如下图所示（Double DQN的形式）：

![optimal bellman-equation](./2-1.png)

在基于深度学习的强化学习中，我们一般将等式右端项称为 ”target label“，而等式左侧为 “基于神经网络的Q-value function”，贝尔曼方程的迭代求解过程即为不断用左侧的 ”神经网络“去逼近右侧新产生的 ”target label“的过程。

在一次最优贝尔曼迭代中，同时涉及到了**evaluation** 和 **improvement** 2个过程：

  （1）**improvement** ，即等式右侧 “新的target label”的计算，主要涉及到对 $\text{argmax}_\vec{a'}Q_\text{joint}(\vec{s'},\vec{a'})$的求解，与单智能体相比，这里最大的挑战在于多智能体的联合动作空间随着智能体的数量呈指数增长，暴力枚举不切实际。针对这一点，设计算法时要考虑 ”${\color{red}\text{Q值函数是否容易求解最大值？}}$“

  （2）**evaluation**，即等式左侧 ”神经网络“对右侧 ”新的target label“进行拟合，设计算法时要考虑 ”${\color{red}\text{基于神经网络的Q值函数是否有足够的拟合能力？}}$“



### Q值函数 “拟合能力” 与 “是否容易求解最大值” 之间的矛盾

函数结构越复杂，其拟合能力会越强，同时导致该函数 “最大值“ 点的求解越困难。

为了更容易求解 ”Q值函数的最大值“，[3] 提出了 **IGM**（Individual-Global-Max）principle：
$$
\forall \tau \in \mathcal{T}, \underset{\boldsymbol{a} \in \mathcal{A}}{\arg \max } Q_\text{joint}(\boldsymbol{\tau}, \boldsymbol{a})=\left(\underset{a^{1} \in \mathcal{A}}{\arg \max } Q_{1}\left(\tau^{1}, a^{1}\right), \ldots, \underset{a^{n} \in \mathcal{A}}{\arg \max } Q_{n}\left(\tau^{n}, a^{n}\right)\right)
$$
即要求各个agents根据individual Q functions 选择的local greedy action构成的joint action 等价于 根据$Q_\text{joint}$ function 选择出的 joint greedy action.



为了方便的满足IGM condition，以VDN [1] 和 QMIX [2] 为例，采用了如下图所示的monotonic mixing网络的设计，即$\forall \mathrm{i}, \frac{\partial \mathbf{Q}_{\text{joint}}(\overrightarrow{\mathbf{s}}, \overrightarrow{\mathbf{a}})}{\partial \mathrm{Q}_{\mathrm{i}}\left(\mathrm{s}_{\mathrm{i}}, \mathrm{a}_{\mathrm{i}}\right)} \geq 0$。于此同时，monotonic的网络结构也限制了$Q_\text{joint}$值函数的拟合能力，详细内容在下一节展开。

![monotonic_nn](C:\Users\15122\OneDrive\论文\【0】article of me\知乎\MARL-7\monotonic_nn.png)

## 3. 算法回顾

### 3.1 符号说明

![notation](C:\Users\15122\OneDrive\论文\【0】article of me\知乎\MARL-7\notation.png)

在后文中，我们统一用$\mathbf{Q}_{\text{joint}}(\overrightarrow{\mathbf{s}}, \overrightarrow{\mathbf{a}})$ 来表示ground truth value（真值），$\mathbf{Q}_{\text{tot}}(\overrightarrow{\mathbf{s}}, \overrightarrow{\mathbf{a}})$来表示拟合值（预估值）。



### 3.2 VDN [1] / QMIX [2]

#### VDN/QMIX的设计

VDN [1] 和 QMIX [2] 的预估值 $\mathbf{Q}_{\text{tot}}(\overrightarrow{\mathbf{s}}, \overrightarrow{\mathbf{a}})$ 采用了如下图所示的monotonic mixing网络的设计：

![monotonic_Q_tot](C:\Users\15122\OneDrive\论文\【0】article of me\知乎\MARL-7\monotonic_Q_tot.png)

![vdn_qmix](C:\Users\15122\OneDrive\论文\【0】article of me\知乎\MARL-7\vdn_qmix.png)

#### 拟合能力受限导致“evaluation”不准确，进一步导致“improvement”陷入次优

为了保证在任意状态下，根据预估值$\mathbf{Q}_{\text{tot}}(\overrightarrow{\mathbf{s}}, \overrightarrow{\mathbf{a}})$选择的最优joint action与根据ground truth $\mathbf{Q}_{\text{joint}}(\overrightarrow{\mathbf{s}}, \overrightarrow{\mathbf{a}})$ 选择出的最优joint action一致，VDN/QMIX 在evaluation时，要求$\forall \overrightarrow{\mathbf{s}},\ \overrightarrow{\mathbf{a}}, \  \mathbf{Q}_{\text{tot}}(\overrightarrow{\mathbf{s}}, \overrightarrow{\mathbf{a}})==\mathbf{Q}_{\text{joint}}(\overrightarrow{\mathbf{s}}, \overrightarrow{\mathbf{a}})$。示意图如上图中右侧所示，横轴为joint action，纵轴为$\mathbf{Q}$ 值大小。可知当“预估值分布”与“真值分布” “完美贴合” 时，可以保证$\mathbf{Q}_{\text{tot}}(\overrightarrow{\mathbf{s}}, \overrightarrow{\mathbf{a}})$与$\mathbf{Q}_{\text{joint}}(\overrightarrow{\mathbf{s}}, \overrightarrow{\mathbf{a}})$ 最大值点一定相同；但是，由于$\mathbf{Q}_{\text{tot}}(\overrightarrow{\mathbf{s}}, \overrightarrow{\mathbf{a}})$网络monotonic的约束，其拟合能力受到很大限制，而这种限制会在某些Return分布下的game中使“预估值分布”与“真值分布” 无法“完美贴合” ，在这种情况下，最大值点就可能发生偏差，从而导致决策 “出错”（evaluation不准确导致improvement非最优）。

#### 举个“出问题”的例子

我们以2个agents的single state matrix game为例（payoff 矩阵如下图），分析monotonic 约束导致的$Q_i$在拟合过程中存在的矛盾性。

![matrix_game](C:\Users\15122\OneDrive\论文\【0】article of me\知乎\MARL-7\matrix_game.png)

记横向、列向 agents 分别为agent-1和agent-2，2个agents均从random policy开始学习。由payoff矩阵可知，情况1：当agent-1选择action $a_1$，agent-2分别选择$b_1$和$b_2$时，有$Q_{\text {tot}}\left(a_{1}, b_{1}\right)>Q_\text{tot}\left(a_{1}, b_{2}\right)$ ($8>-12$）。由于agent-1输入$\mathbf{Q}_{\text{tot}}$网络部分为$Q_{1}\left(a_{1}\right)$ 不变，根据$\mathbf{Q}_{\text{tot}}(\overrightarrow{\mathbf{s}}, \overrightarrow{\mathbf{a}})$网络的单调性 $\Rightarrow$ 对agent-2有 $ Q_{2}\left(b_{1}\right)>Q_{2}\left(b_{2}\right)$。

同理，情况2：当agent-1选择action $a_2$，agent-2分别选择$b_1$和$b_2$时，有$Q_{\text {tot}}\left(a_{2}, b_{1}\right)<Q_\text{tot}\left(a_{2}, b_{2}\right)$ ($-12<0$）， 根据$\mathbf{Q}_{\text{tot}}(\overrightarrow{\mathbf{s}}, \overrightarrow{\mathbf{a}})$网络的单调性$\Rightarrow$ 对agent-2有 $ Q_{2}\left(b_{1}\right)<Q_{2}\left(b_{2}\right)$，与情况1矛盾，导致$Q_i$无准确解。

#### 线性方程组视角看VDN

以VDN为例，上述拟合问题本质是线性方程组的求解，由于2个agents均采取random policies，9条样本出现频率相同，对应的线性方程组如下图所示：

![linear_equation](C:\Users\15122\OneDrive\论文\【0】article of me\知乎\MARL-7\linear_equation.png)

很容易知道此方程组无解。根据最小二乘法，有，$Q_i=(A^\mathsf{T}A)^{-1}A^\mathsf{T}y$，这里$A$为系数矩阵，$y$为label (payoff)，在这个例子里$A^\mathsf{T}A$恰好为singular matrix，不可逆，加入L2正则项，解析解变为$Q_i=(A^\mathsf{T}A+\lambda E)^{-1}A^\mathsf{T}y$

求解如下：

```python
import numpy as np
A = np.array([
    [1, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1],
    [0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 1],
], dtype=np.float)
y = np.array([8, -12, -12, -12, 0, 0, -12, 0, 0], np.float).reshape([-1, 1])
B = np.matmul(A.T, A)
E = np.eye(*B.shape)
_lambda = 0.00000001
q_i = np.matmul(np.matmul(np.linalg.inv(B + _lambda * E), A.T), y)
print(q_i)
```

结果为：

```python
[[-3.11111113]  <--Q1(a_1)
 [-1.77777781]  <--Q1(a_2)
 [-1.77777781]  <--Q1(a_3)
 [-3.11111107]  <--Q2(b_1)
 [-1.77777776]  <--Q2(b_2)
 [-1.77777766]] <--Q2(b_3)
```

由结果知，$Q_1(a_1)$和$Q_2(b_1)$数值均小于各自其他2个action，因此，individual的最优action非全局最优。这就是$\mathbf{Q}_{\text{tot}}$网络拟合（表征）能力受限，导致估值不准，从而导致决策出错。

#### 举个“没问题”的例子

仍2个agents的single state matrix game为例，payoff 矩阵如下图：

![matrix_game2](C:\Users\15122\OneDrive\论文\【0】article of me\知乎\MARL-7\matrix_game2.png)

 可知，不管横向agent-1，选择action $a_1, a_2$还是$a_3$，对于列向agent-2而言，始终有$Q_{2}\left(b_{1}\right)>Q_{2}\left(b_{2}\right)>Q_{2}\left(b_{3}\right)$。同理，对列向agent-2也一样，不管其选择action $b_1, b_2$还是$b_3$，都不会影响agent-1 各个action individual Q值的相对大小，即$Q_{1}\left(a_{1}\right)>Q_{1}\left(a_{2}\right)==Q_{1}\left(a_{3}\right)$。因此，各自的最优action始终为$a_1$和$b_1$，与全局最优保持一致。

同样，从线性方程组的视角来看：

```python
import numpy as np
A = np.array([
    [1, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1],
    [0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 1],
], dtype=np.float)
y = np.array([8, 3, 2, -12, -13, -14, -12, -13, -14], np.float).reshape([-1, 1])
B = np.matmul(A.T, A)
E = np.eye(*B.shape)
_lambda = 0.00000001
q_i = np.matmul(np.matmul(np.linalg.inv(B + _lambda * E), A.T), y)
print(q_i)
```

结果为：

```
[[ 7.9444444 ]  <--Q1(a_1)
 [-9.38888887]  <--Q1(a_2)
 [-9.38888887]  <--Q1(a_3)
 [-1.72222219]  <--Q2(b_1)
 [-4.05555557]  <--Q2(b_2)
 [-5.05555539]] <--Q2(b_3)
```

由结果只，$Q_1(a_1)$和$Q_2(b_1)$数值均大于各自其他2个action，因此，individual的最优action即是全局最优。这时，虽然$\mathbf{Q}_{\text{tot}}$网络拟合（表征）能力受限，无法做到精准拟合，但是由于2个agents各个action individual Q值大小的“偏序关系”不随另一个agent所采取的动作变化而改变，因此决策不受影响。



#### 针对这两个例子QMIX的结果

与VDN相比，QMIX的$\mathbf{Q}_{\text{tot}}$不再是“对各个agents的individual Qs”进行简单的线性加和，而是经过了“更复杂的monotonic mixing网络的非线性变化”，因此，无法直接求得其解析解。这里我们用梯度下降的优化方式，来求解QMIX对上述两个例子的拟合结果，结果如下：

##### “出问题”的例子

```python
Iter=1400: MSE loss=35.56791687011719
Iter=1500: MSE loss=35.5675163269043
Iter=1600: MSE loss=35.56715393066406
Iter=1700: MSE loss=35.56682205200195
Iter=1800: MSE loss=35.56650161743164
Iter=1900: MSE loss=35.56621551513672
******************* Individual q tables *******************
-------------- agent-0: greedy action=2 --------------
[-7.55502462387085, 0.5389440059661865, 0.5389440059661865]
--------------------------------------
-------------- agent-1: greedy action=2 --------------
[-5.721792697906494, 0.11790931969881058, 0.11790931969881058]
--------------------------------------
******************* Predicted Q_tot: *******************
[[-7.992419719696045], [-7.992414951324463], [-7.992414951324463]]
[[-7.99241304397583], [0.15237689018249512], [0.15237689018249512]]
[[-7.99241304397583], [0.15237689018249512], [0.15237689018249512]]
******************* True Q_joint: *******************
[[8.0], [-12.0], [-12.0]]
[[-12.0], [0.0], [0.0]]
[[-12.0], [0.0], [0.0]]
```

##### “没问题”的例子

```
Iter=1400: MSE loss=0.14740869402885437
Iter=1500: MSE loss=0.14264029264450073
Iter=1600: MSE loss=0.14104598760604858
Iter=1700: MSE loss=0.13940799236297607
Iter=1800: MSE loss=0.1388709396123886
Iter=1900: MSE loss=0.13802725076675415
******************* Individual q tables *******************
-------------- agent-0: greedy action=0 --------------
[2.2560274600982666, 1.0452449321746826, 1.0452449321746826]
--------------------------------------
-------------- agent-1: greedy action=0 --------------
[-0.840924859046936, -1.1037302017211914, -1.1742500066757202]
--------------------------------------
******************* Predicted Q_tot: *******************
[[7.5554704666137695], [2.7076873779296875], [1.406857967376709]]
[[-12.115528106689453], [-13.439348220825195], [-13.692365646362305]]
[[-12.115528106689453], [-13.439348220825195], [-13.692365646362305]]
******************* True Q_joint: *******************
[[8.0], [3.0], [2.0]]
[[-12.0], [-13.0], [-14.0]]
[[-12.0], [-13.0], [-14.0]]
```

从上面结果可以看出，QMIX预估的$Q_\text{tot}$相比VDN更加准确些，但在第一个例子中，由于各个action individual Q值大小的“偏序关系”会随另一个agent所采取的动作变化而改变，收敛到MSE最低时，决策仍为“次优解”。



### 3.3 QTRAN [3]

#### QTRAN的设计

回顾VDN/QMIX，为了保证在任意状态下，根据预估值$\mathbf{Q}_{\text{tot}}(\overrightarrow{\mathbf{s}}, \overrightarrow{\mathbf{a}})$选择的最优joint action与根据ground truth $\mathbf{Q}_{\text{joint}}(\overrightarrow{\mathbf{s}}, \overrightarrow{\mathbf{a}})$ 选择出的最优joint action一致，VDN/QMIX 在evaluation时，要求$\forall \overrightarrow{\mathbf{s}},\ \overrightarrow{\mathbf{a}}, \  \mathbf{Q}_{\text{tot}}(\overrightarrow{\mathbf{s}}, \overrightarrow{\mathbf{a}})==\mathbf{Q}_{\text{joint}}(\overrightarrow{\mathbf{s}}, \overrightarrow{\mathbf{a}})$。与之相比，QTRAN放宽了限制，可以看做是一种“把好钢用在刀刃上”的做法。

![vdn_style_qtran](C:\Users\15122\OneDrive\论文\【0】article of me\知乎\MARL-7\vdn_style_qtran.png)

具体，我们这里对比 VDN 与 VDN style QTRAN （VDN style是指$\mathbf{Q}_{\text{tot}}$采用对individual $Q_i$线性加和的形式）的区别。如上图所示，VDN要求$\forall \overrightarrow{\mathbf{s}},\ \overrightarrow{\mathbf{a}}, \  \mathbf{Q}_{\text{tot}}(\overrightarrow{\mathbf{s}}, \overrightarrow{\mathbf{a}})==\mathbf{Q}_{\text{joint}}(\overrightarrow{\mathbf{s}}, \overrightarrow{\mathbf{a}})$，而QTRAN做了分情况讨论：（1）当所有agents都采取individual greedy action时，要保证 $\mathbf{Q}_{\text{tot}}==\mathbf{Q}_{\text{joint}}$；（2）对其他所有情况（即任意agent没有采取individual greedy action时），只需要 $\mathbf{Q}_{\text{tot}}\ge\mathbf{Q}_{\text{joint}}$，也就是$\mathbf{Q}_{\text{tot}}$的拟合能力虽然受限，但放宽了约束条件，不要求“严格取等”，只需要“大于等于”。

两者对比示意图如下图所示，VDN（如下图左）要求“严丝合缝”贴合（存在因拟合能力受限而无法做到的情况）。而QTRAN（如下图右）只要求$\mathbf{Q}_{\text{tot}}$去拟合$\mathbf{Q}_{\text{joint}}$的一个上包络线，在预估值$\mathbf{Q}_{\text{tot}}$的“最大值处“要保持与真值$\mathbf{Q}_{\text{joint}}$“严格相等”，其他所有地方 $\mathbf{Q}_{\text{tot}}\ge\mathbf{Q}_{\text{joint}}$，满足了这2点，则能保证2条曲线只在最大值点重合。

由于$\mathbf{Q}_{\text{tot}}$的monotonicity，各个agents根据individual $Q_i$ 选择的local greedy action即为$\mathbf{Q}_{\text{tot}}$的最大值点，同时由于2条曲线只在最大值点重合，$\mathbf{Q}_{\text{tot}}$的最大值点同时也是$Q_\text{joint}$的最大值点，因此满足了IGM condition（各个agents根据individual $Q_i$ 选择的local greedy action构成的joint action 等价于 根据$Q_\text{joint}$ 选择出的 joint greedy action）。

![vdn_vs_qtran](C:\Users\15122\OneDrive\论文\【0】article of me\知乎\MARL-7\vdn_vs_qtran.png)

于此同时，QTRAN中的$\mathbf{Q}_{\text{tot}}$不再要求要在所有joint actions处与$\mathbf{Q}_{\text{joint}}$“严丝合缝的贴合”，因此尽管$\mathbf{Q}_{\text{tot}}$拟合能力受限，QTRAN仍能收敛至最优解。

#### 对于VDN/QMIX 出问题的例子

![matrix_game](C:\Users\15122\OneDrive\论文\【0】article of me\知乎\MARL-7\matrix_game.png)

```
Iter=1400: QTRAN loss=0.0
Iter=1500: QTRAN loss=0.0
Iter=1600: QTRAN loss=0.0
Iter=1700: QTRAN loss=0.0
Iter=1800: QTRAN loss=0.0
Iter=1900: QTRAN loss=0.0
******************* [q_i] Individual q tables *******************
-------------- agent-0: greedy action=0 --------------
[4.082147121429443, 0.08670942485332489, 0.08740530908107758]
--------------------------------------

-------------- agent-1: greedy action=0 --------------
[3.9178526401519775, -0.08670942485332489, -0.02233865298330784]
--------------------------------------

******************* Predicted Q_tot: *******************
[[8.0], [3.9954376220703125], [4.059808254241943]]
[[4.004561901092529], [0.0], [0.0643707737326622]]
[[4.005258083343506], [0.0006958842277526855], [0.06506665796041489]]

******************* True Q_joint: *******************
[[8.0], [-12.0], [-12.0]]
[[-12.0], [0.0], [0.0]]
[[-12.0], [0.0], [0.0]]
```

可以看到，在最优点 action ($a_1$, $b_1$)处，$Q_\text{tot}=Q_\text{joint}$，在其他actions处$Q_\text{tot}\ge Q_\text{joint}$。此外QTRAN中的$Q_\text{tot}$除了可以采用VDN的形式，也可以采用QMIX中引入多层monotonic mixing网络的形式。我们这里没有介绍实现上的更多细节，比如$Q_\text{joint}$采用额外一个没有monotonicity约束的Q-function来提供，详细实现可以参考原文 [3]。

#### QTRAN的缺点："$\ge$"约束优化困难

QTRAN要求（1）当所有agents都采取individual greedy action时，要保证 $\mathbf{Q}_{\text{tot}}==\mathbf{Q}_{\text{joint}}$；（2）对其他**所有情况**（即任意agent没有采取individual greedy action时），要求 $\mathbf{Q}_{\text{tot}}\ge\mathbf{Q}_{\text{joint}}$。

类比contrastive learning，把（1）看做正样本采样，（2）看做负样本采样， 对于（2），在每个状态下，要确保在其他**全部$|A|-1$个**joint actions负样本点“$\ge$”都要成立，这需要大量的采样。而实际优化过程中，难以做到这一点。

我们举个因”负样本不足“而导致的陷入次优的情况：

![qtran_insufficient_sample](C:\Users\15122\OneDrive\论文\【0】article of me\知乎\MARL-7\qtran_insufficient_sample.png)

两条绿色“竖线”中间部分对应的joint action点，因训练过程中负样本点采样不足而被“忽略”，导致预估网络$\mathbf{Q}_{\text{tot}}$在“见到”过的样本中的最优非“全局最优”。



### 3.4 QPLEX [7]

#### QPLEX是借鉴了QTRAN思想的另一种“实现”

![qplex](C:\Users\15122\OneDrive\论文\【0】article of me\知乎\MARL-7\qplex.png)

QPLEX预估网络$\mathbf{Q}_{\text{tot}}$设计的核心依然围绕“在方便求解最大值”的同时“增强网络的表征能力”展开，核心为 ${\color{red}Q_\text{tot}^\text{max}-\text{difference}}$的设计，构造的等式如上图所示。与QTRAN的2个条件非常相似：

（1）当所有agents都采取individual greedy action时， $\mathbf{Q}_{\text{tot}}=Q_1(\bar a_1)+Q_2(\bar a_2)$，要求其$==\mathbf{Q}_{\text{joint}}$；

（2）对其他所有情况（即任意agent没有采取individual greedy action时），$\mathbf{Q}_{\text{tot}}=Q_1(\bar a_1)+Q_2(\bar a_2)-\text{Difference}(\vec s,\left<a_1, a_2\right>)==\mathbf{Q}_{\text{joint}}$。

$\text{Difference}(\vec s,\left<a_1, a_2\right>)$函数具体定义为：$\lambda(\vec{s}, \vec{a})\left\{\left[Q_{1}\left(\bar{a}_{1}\right)-Q_{1}\left(a_{1}\right)\right]+\left[Q_{2}\left(\bar{a}_{2}\right)-Q_{2}\left(a_{2}\right)\right]\right\}$，其中（1）前一项$\lambda(\vec{s}, \vec{a})\gt0$，且因为其以完整的joint state和joint action作为输入，且没有其他任何约束，根据universal function approximation of neural networks，其可以逼近任意$\gt0$的函数。（2）后一项$\left\{\left[Q_{1}\left(\bar{a}_{1}\right)-Q_{1}\left(a_{1}\right)\right]+\left[Q_{2}\left(\bar{a}_{2}\right)-Q_{2}\left(a_{2}\right)\right]\right\}\ge 0$当且仅当所有agents都采取individual greedy action时取到0。结合2项，$\text{Difference}(\vec s,\left<a_1, a_2\right>)$为没有任何约束的函数，具备任意的拟合能力，用其来拟合$\mathbf{Q}_{\text{joint}}$最大值与任意$\mathbf{Q}_{\text{joint}}$值之间的差距（此差距$\ge0$）。



#### 对于VDN/QMIX 出问题的例子

![matrix_game](C:\Users\15122\OneDrive\论文\【0】article of me\知乎\MARL-7\matrix_game.png)

```
Iter=1500: MSE loss=0.1864847093820572
Iter=1600: MSE loss=0.1817486435174942
Iter=1700: MSE loss=0.1469857096672058
Iter=1800: MSE loss=0.12402277439832687
Iter=1900: MSE loss=0.1674421727657318
******************* [q_i] Learned individual q tables *******************
-------------- agent-0: greedy action=0 --------------
[3.2727248668670654, 0.2756689190864563, -0.36994096636772156]
--------------------------------------

-------------- agent-1: greedy action=0 --------------
[4.60710334777832, 0.8946372270584106, 1.139521598815918]
--------------------------------------

******************* Predicted Q_tot: *******************
[[7.879828453063965], [-12.263760566711426], [-12.260972023010254]]
[[-11.70386028289795], [0.5823078155517578], [0.5869455337524414]]
[[-11.68749713897705], [0.4184398651123047], [0.4870738983154297]]

******************* True Q_joint: *******************
[[8.0], [-12.0], [-12.0]]
[[-12.0], [0.0], [0.0]]
[[-12.0], [0.0], [0.0]]
```

可以看到，在一般情况下evaluation比较准确，可以收敛至最优解。



#### QPLEX的缺点：容易陷入“局部最优”

![qplex_sub_optimal](C:\Users\15122\OneDrive\论文\【0】article of me\知乎\MARL-7\qplex_sub_optimal.png)

QPLEX缺陷来自于其核心${\color{red}Q_\text{tot}^\text{max}-\text{difference}}$的设计，即（1）首先把当前$Q_\text{tot}$预估的最大值锚定为整个函数空间的上界；（2）其他任意值表示为此锚定值减去一个大于0的差值。

以上图图（a）为例，在优化初始阶段，锚定值“不正确”（也就是$Q_\text{tot}$的最大值$\neq$$Q_\text{joint}$的最大值）， $Q_\text{tot}$在此锚定的上界约束下，其他位置Q值均小于该锚定值。此时，$Q_\text{tot}$的拟合Error很大，要想缩小拟合Error，锚定点右侧的红色虚线要对应的拉高（如红色箭头所示），以此来贴合蓝色的真值曲线。

然而，由于锚定点的存在，difference最小为0，也就是在锚定点保持不动的条件下，只能优化到如上图（b）所示的结果。此时，若想进一步减少拟合Error，必须把锚定点的Q值也进一步拉高。

进一步，将锚定点拉高必然导致如上图（c）所示的结果，锚定点处（以及左侧部分）因向上移动偏离了原来的真值位置，受到向下的阻力，而锚定点右侧为了进一步减少拟合Error需要进一步向上拉伸。因此，锚定点处同时受到一上一下2个互相矛盾的力，优化出现矛盾，最终，只会收敛到2力平衡位置，而收敛到的解未必一定为最优解。

#### 还是这个例子

![matrix_game](C:\Users\15122\OneDrive\论文\【0】article of me\知乎\MARL-7\matrix_game.png)

```
Iter=1500: MSE loss=3.6293046474456787
Iter=1600: MSE loss=3.6290104389190674
Iter=1700: MSE loss=3.6287240982055664
Iter=1800: MSE loss=3.628448486328125
Iter=1900: MSE loss=3.628178119659424
******************* [q_i] Learned individual q tables *******************
-------------- agent-0: greedy action=2 --------------
[-1.1775593757629395, 1.2898921966552734, 2.025874376296997]
--------------------------------------

-------------- agent-1: greedy action=2 --------------
[-1.1501485109329224, 1.3891468048095703, 1.9467390775680542]
--------------------------------------

******************* Predicted Q_tot: *******************
[[3.9428420066833496], [-11.683112144470215], [-11.747499465942383]]
[[-11.682787895202637], [0.21431350708007812], [0.13855814933776855]]
[[-11.750524520874023], [0.13001537322998047], [3.9726133346557617]]

******************* True Q_joint: *******************
[[8.0], [-12.0], [-12.0]]
[[-12.0], [0.0], [0.0]]
[[-12.0], [0.0], [0.0]]
```

由以上结果可以看到，在很多情况下，再上述2个矛盾力量的影响下，QPLEX未必可以收敛到MSE loss为0（全局最优）处，此时优化卡在了次优点，evaluation的精度无法再进一步提升，同时决策也非全局最优解。

### 3.5 其他相关算法

本文未介绍的Weighted QMIX[5], QTRAN++ [6]等算法，核心思想与QTRAN、QPLEX十分相似，本次暂不做更详细介绍。



## 4. 小结

上述几个工作都是基于multiagent centralized Q-learning，围绕“在方便求解最大值”的同时“增强网络的表征能力”展开，在方法设计上有一定的创新性。总的来说，网络结构设计越复杂，其表征能力会越强，但与此同时，训练所需要的样本数量也随之增加，收敛变得更慢也更困难。

最后谈一点个人感受：目前multiagent这一块可以说是“百花齐放（乱七八糟）“，A说A是SOTA，B说B是SOTA，反正大家就都是SOTA，其乐融融，好不热闹。没有合理的对比实验，只是为了“绩效”而不负责任地刷文章的话，就成了“故事会”。对于MA实验效果的置信度问题（尤其是在SMAC [7]上的工作），可以参考另一篇优秀的博文：[《多智能体强化学习实验打脸集合》](https://zhuanlan.zhihu.com/p/408515796)。

目前，前路仍然是黑夜，看不清MA何去何从。不过，努力让自己的工作越来越solid吧，期待拨云见日的一天。

![taiji-02](C:\Users\15122\OneDrive\论文\【0】article of me\知乎\MARL-7\太极张三丰\taiji-02.jpg)



## 参考

[1] Sunehag P, Lever G, Gruslys A, et al. Value-Decomposition Networks For Cooperative Multi-Agent Learning Based On Team Reward[C]//Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems. 2018: 2085-2087.

[2] [QMIX] Rashid T, Samvelyan M, Schroeder C, et al. Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning[C]//International Conference on Machine Learning. PMLR, 2018: 4295-4304.

[3] [QTRAN] Son K, Kim D, Kang W J, et al. Qtran: Learning to factorize with transformation for cooperative multi-agent reinforcement learning[C]//International Conference on Machine Learning. PMLR, 2019: 5887-5896.

[4] [Qatten] Yang Y, Hao J, Liao B, et al. Qatten: A general framework for cooperative multiagent reinforcement learning[J]. arXiv preprint arXiv:2002.03939, 2020.

[5] [Weighted-QMIX] Rashid T, Farquhar G, Peng B, et al. Weighted QMIX: Expanding Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning[J]. Advances in Neural Information Processing Systems, 2020, 33.

[6] [QTRAN++] Son K, Ahn S, Reyes R D, et al. QOPT: Optimistic Value Function Decentralization for Cooperative Multi-Agent Reinforcement Learning[J]. arXiv preprint arXiv:2006.12010, 2020.

[7] [QPLEX] Wang J, Ren Z, Liu T, et al. Qplex: Duplex dueling multi-agent q-learning[J]. arXiv preprint arXiv:2008.01062, 2020.

[8] [SMAC] Samvelyan M, Rashid T, De Witt C S, et al. The starcraft multi-agent challenge[J]. arXiv preprint arXiv:1902.04043, 2019.
