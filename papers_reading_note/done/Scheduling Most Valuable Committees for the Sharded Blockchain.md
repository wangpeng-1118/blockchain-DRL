# Abstract

在分片区块链中，交易由一系列平行的committee处理，交易吞吐量因此大量提升。但是在一个epoch的开始阶段，形成committee有很大的延迟，不同节点的异构处理能力导致不平衡的共识延迟。需要一个好的committee-scheduling策略来减少交易的cumulative age。

研究内容：大规模区块链中，交易的吞吐量和cumulative age之间的精细平衡(fine-balanced tradeoff between the transactions’ throughput and their cumulative age in a large-scale sharded blockchain)

算法：online distributed Stochastic-Exploration(SE) algorithm

# Introduction

Nakamoto consensus：比特币中的去中心化共识机制包括工作量证明，最长链规则

Elastico:将区块链网络拆分成更小的由一组矿工组成的committee，矿工协作处理不相交的交易集

Elastico步骤：

1、矿工通过PoW选择机制选择committee

2、矿工通过互相交换committee成员信息来确认和发现彼此

3、矿工通过运行拜占庭协议达成committee内部的共识

4、所有committee产生的分片被提交给final committee，final committee为root chain产生一个全新的全局块

5、final committee产生一系列随机字符串用于下一个epoch产生新的committee

Fig.2(b)中CDF表示，在一定时间内被确认的交易的比例

在Elastico中，每个committee的deadline被定义为个人的独立分片被提到到final committee的时间

cumulative age：committee从形成阶段到deadline的总等待时间

# The Related Work

RapidChain：通过利用一种有效的跨分片验证方法来避免泛洪广播消息，从而提高了吞吐量

SharPer：一种许可区块链系统，用于通过将不同的数据分片划分和重新分配到各种网络集群来提高区块链的可扩展性

OptChain：不同于随机分片的新分片放置范式，可以最小化跨分片事务的数量

最新的分片协议通过提出新的处理跨分片交易的方法或者通过转移账户以减少大量的跨分片交易来提升可拓展性或者吞吐量

本文尝试通过优化成员committee执行final consensus时不平衡的两阶段延迟来提升分片区块链中的吞吐量

# Problem Statement

假设分片协议为每个member committee配置一个DDL

在一个特定的DDL，final committee需要从所有member committee中产生的分片中选择一部分，通过运行拜占庭协议产生一个final block

本文提出的方法旨在帮助final committee在DDL到达之前选择参与final consensus的member committee

实践中将DDL设置为预定百分比的committee将自己的分片提交给final committee

最大化每个epoch的接收的交易数量，最小化cumulative age

MVCom(the Most Valuable Committees)

# Online Distributed Stochastic-Exploration Algorithm

这种基于SE的算法能够实时做出决策，通过剔除两个阶段的延迟超过预设截止时间（DDL）的慢节点，以及剔除价值较低的一组分片，从而终止当前的训练轮次。

所提出的 SE 算法可以在分布式方式下，独立于分片区块链系统之外的某个地方执行。

在所实现的马尔可夫链中，如果状态之间的转移能够被训练为收敛到期望的平稳分布p ∈ f，则系统可以实现接近最优的性能

























