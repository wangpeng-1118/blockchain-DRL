# Abstract

LLM在生成长文本时，进行 LLM 推理服务会面临挑战：模型的中间状态（即 KV 缓存）占用的内存非常大，并且随着序列长度和批大小的增加而线性增长。

InfiniGen:一个专为长文本生成设计的全新 KV 缓存管理框架，可与现代的**基于卸载（offloading）机制的推理系统**协同工作

key insight of InfiniGen:用于计算 Transformer 下一层注意力的关键 token，可以通过在当前层输入上执行==最小化“预演”操作(minimal rehearsal)==，并结合下一层部分查询权重与 KV 缓存内容进行推测得到。

系统只需预取关键的 KV 缓存项，而无需将全部缓存加载

# Introduction

随着用户对更长序列和更大 batch size 的需求不断增长，KV 缓存带来的内存压力将变得更加突出。

InfiniGen:为长文本生成设计的KV缓存管理框架，旨在与现代卸载式推理系统协同工作。

两个设计理念：

* 通过在第 i 层中对注意力计算进行最小化预演，从而推测出对生成下一个 token 起关键作用的 KV 缓存项，并丢弃非关键项。
* 利用 CPU 大容量内存，在 CPU 中维护 KV 缓存池，从而确保在生成长文本时始终能够识别出关键的 KV 值，动态移除不经常使用的标记的KV Cache。

特别是，InfiniGen通过离线操作模型权重，使推测更加高效和精确，==通过倾斜Transformer架构的查询和键矩阵，强调某些重要的列。==

在预填充阶段，当推理请求的提示和输入最初被处理时，InfiniGen会生成后续解码阶段（即输出生成阶段）使用的部分权重。在解码阶段的第i - 1层，InfiniGen使用第i - 1层的注意力输入、部分查询权重和第i层的部分键缓存来推测下一层（第i层）的注意力模式。

contribution：

* 提出了InfiniGen，这是一个动态KV缓存管理框架，通过智能管理CPU内存中的KV缓存，与现代基于卸载的LLM服务系统协同工作
* 提出了一种新的KV缓存预取技术与短暂修剪，推测后续的注意力层的注意力模式，并带来了KV缓存的主要部分，而在CPU内存中保留其余部分
* 我们在一个现代的基于卸载的推理系统上实现了InfiniGen，并证明它大大优于现有的KV缓存管理方法，实现了高达3.00倍的性能，同时还提供了更好的模型精度。

# background

输入张量的维度N * D,N为输入的token的数量，D是张量的维度（模型的维度）。先进行层归一化，再输入到注意力层。通过不同的权重矩阵，每个token对应生成Q，K，V。矩阵被重塑为H * D * d，H是注意头的数量，d是头的维度，D = H * d。

**outlier**：指某些参数值（或激活值）特别大或特别小，远离平均值，跟其他值差很多的“异常数值”。可能出现在weights和activations中，在低比特量化中，是精度损失的主要来源。

==奇异值分解==

#  Motivation

现代LLM服务系统，如DeepSpeed和FlexGen，已经支持将模型权重或KV缓存卸载到CPU内存。当涉及到基于卸载的推理系统时，KV缓存大小变得更加成问题，这是由于CPU和GPU之间的低PCIe带宽，它成为了新的关键瓶颈

尽管通过量化压缩KV缓存可以潜在地减少基于卸载的系统中的数据传输开销，但这并不是一个根本性的解决方案，因为量化并没有解决KV缓存问题的根本原因，即KV条目随着序列长度的线性增长

**Challenges in KV Cache Management**: 减轻KV缓存从CPU到GPU的传输开销的根本方法是通过识别计算注意力分数的关键键和值来减少要加载的KV缓存的量

先前的关于管理KV缓存的工作并不能有效地解决基于卸载的推理系统中的以下挑战

* Dynamic nature of attention patterns across iterations(迭代间注意力模式的动态性)：在当前迭代中被认为不重要的标记可能在后续迭代中变得重要。因此，H2O在序列长度超过KV缓存预算时开始与动态注意力模式作斗争，导致余弦相似度低于Optimal情况
* **Adjusting the number of KV entries across layers（跨层调整KV条目数量）**:不同层对KV缓存的需求不同，我们需要跨不同层动态调整参与注意力计算的关键标记数量，以有效利用KV缓存预算
* **Adjusting the number of KV entries across queries（跨查询调整KV条目数量）**：

# InfiniGen Design

## Overview

InfiniGen的核心设计理念是利用丰富的CPU内存容量，以增加在KV缓存中识别重要token的窗口大小。

并不会将整个KV Cach保存在GPU中，二十仅加载少数几个重要token的key和value，动态地丢弃其他不重要的KV Cache。

InfiniGen 不是直接通过列求和找 token，而是通过列求和找“最重要的维度（列）”，然后**再在这些维度上看哪些 token（行）值最大**，从而判断最关键的 token。

因为 Transformer 的 Attention 是**点积**，如果某些维度（列）在 Q 和 K 中都很大，就会对 Attention 分数贡献非常大；

只使用上面的方法求出来的列进行attention权重的计算，然后在decode阶段选择权重位于$[max - \alpha, max]$的token进行计算

我们计算
$$
Q' = X \cdot W_Q \cdot V = X \cdot U \Sigma V^T V
$$
注意到：
$$
V^T \cdot V = I
$$
所以上式等价于：
$$
Q' = X \cdot U \Sigma
$$
这就完成了一个“坐标空间旋转”，**把原来的 Q 向量投影到了一个新的坐标系中**，而这个坐标系是：**按照奇异值从大到小排序的方向（重要性排序的方向）**





























