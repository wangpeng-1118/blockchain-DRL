# 比特币

## 密码学原理

输入空间远远大于输出空间，哈希碰撞不可避免

collision：x $ \neq $ y，H(x) = H(y)

**collision resistance**：哈希碰撞避免（没有人为的办法制造哈希碰撞，理论上无法被数学证明）

作用：求一个message的digest（信息摘要），文件被修改之后，它的哈希值也会被修改

**hiding**：从哈希值无法反推出输入，除非遍历所有可能的输入

digital commitment：digital equivalent of a sealed envelope(密封信封的数字等价物,数字签名)

实际中在输入中添加一个nonce，以保证输入的随机性

**Puzzle friendly**：用于比特币里面的PoW

创建用户：建立一对密钥

非对称加密体系：公钥加密，私钥解密

私钥签名，公钥验证

## BTC-数据结构

Block chain is a linked list using hash pointers

Merkle tree（哈希指针代替了普通指针）：只要记住root hash，就能检测出树中任意地方的修改

每个区块所包含的交易组织成一个Merkle tree

Merkle tree 的root hash存在block header中，交易列表存在block body中

![Merble tree](pictures\20250309145445.png)

**比特币的全节点**（Full Node）确实保存了 **整个区块链中的所有区块的全部信息**，并且参与了区块链网络的验证和传播工作。

**轻节点（SPV 节点）**只保存 **当前与自己相关的区块头**，而不是所有区块头。

一个区块头包含：

* `version`：版本号

- `previous block hash`：前一个区块的哈希值
- `merkle root`：该区块的 Merkle 树根哈希
- `timestamp`：区块的时间戳
- `target`：目标难度（用于工作量证明）
- `nonce`：用来调整工作量证明的值

proof of membership:时间复杂度$ O(\log_2(n)) $ 

proof of non-membership:如果叶节点按照哈希值排序之后就只需要验证相关交易前后两个交易，BTC中没有这种验证需求

只要数据结构是无环的，均可以用哈希指针代替普通地址指针

## BTC-协议

double spending sttack

为防范double spending：需要额外的哈希指针指向之前的交易，

例如A给B转账十个bitcoin，A的10个比特币来自铸币交易，那么铸币里面会有A的公钥的哈希，来自A的交易里面的A的公钥要和前面铸币交易里面的A的公钥的哈希对得上

**Distributed hash table**:系统中很多台机器共同维护一个哈希表

FLP impossibility result：asynchronous(异步的，传输时延没有上限) system中，即使只有一个成员有问题（faulty），也无法达成共识

CAP Theorem（consistency， availability， partition tolerance：分区容错性）：任何一个分布式系统最多只可能满足前面提到的两个性质

**consensus in bitcoin**：$H(block header) \leqslant target$ 投票与算力有关

**longest valid chain**:接收的区块应该是接在最长合法链中的

**forking attack**

**block reward**：(coinbase transaction出币交易)

BTC总量:$21W \times 50 \times (1 + \frac{1}{2} \times \frac{1}{4} \dots) = 2100W$  

## BTC-实现

**UTXO(Unspent Transaction Output)**:未被花掉的Bitcoin的交易记录会被保存到UTXO中，用于检测新发布的交易是否合法

**Transaction Fee**

修改铸币交易中Coinbase中的内容，进而改变Merkle Root Hash的值，以达到增加哈希搜索空间的目的

挖矿的每一次尝试都可以看作一次Bernoulli trial（a random experiment with binary outcome)

Bernoulli process : a sequence of independent Bernoulli trials

概率P很小，尝试的次数N很大，Bernoulli process可以使用Poisson process近似

出矿时间服从指数分布，过去的工作对后面出矿的概率没有影响(progress free)

progress free保证了矿工的优势和算力的优势成立比，不会因为之前做的工作多而产生碾压式的优势

Bitcoin is secured by mining：只要大部分的算力掌握在诚实的人的手里，BTC系统的安全性就能得以保证

BTC系统规定，每个区块只能有1MB大小，某个时间段交易数量太多可能会导致一些交易无法及时发布到区块链上

selfish mining:自己挖很多个区块，某个时间段一次性发布（回滚以前的交易，也可以用于浪费其他人的算力）

## BTC-网络

application layer: BitCoin Block chain

network layer: P2P Overlay Network

simple, robust,  but not efficient

means of communication : flooding(unrelated to actual physical distance)

对一个节点来说，已经转发过的交易就不会再转发了

## BTC-挖矿难度

分叉变多，网络中的算力被分散，恶意节点的分叉攻击会变得相对容易

调整target
$$
target = target \times \frac{actual\_time}{expected\_time}
$$
$actual\_time$ 会被设置一个上下限

## BTC-挖矿

密码学和共识机制保证比特币的安全

ASIC：Application-Specific Integrated Circuit

大型矿池：矿主管理许多矿工，挖到矿之后将收益进行分配，矿池的矿工可以使用almost valid block来进行自己的工作证明（没有其他用处）

矿池的危害：攻击者只需要吸引大量不明真相的矿工来为自己挖矿，就可以发起一次进攻

攻击方式：分叉攻击（回滚部分交易），封锁账户攻击（当某个区块包含特定账户的交易时，立即进行分叉攻击）

## BTC-脚本

后面一个交易的input部分和前一个交易的output拼接执行

```BTCscripting
PUSHDATA(sig)      //入栈
PUSHDATA(PubKey)   //入栈
CHECKSIG           //检验
```

**P2PKH(Pay to Public Key Hash)**

```BTCscript
input script:
	PUSHDATA(Sig)
	PUSHDATA(PubKey)
output script:
	DUP						//copy栈顶元素
	HASH160					//取栈顶元素，求hash再压入栈
	PUSHDATA(PubKeyHash)	//让之前账户的收款人的公钥哈希入栈
	EQUALVERIFY				//弹出栈顶的两个元素（公钥哈希）判断是否相等
	CHECKSIG
```

**使用P2SH实现多重签名**

```BTCscript
input acript:
	false							//无意义，用于应付CHECKMULTISIG的bug
	PUSHDATA(Sig_1)
	PUSHDATA(Sig_2)
	...
	PUSHDATA(Sig_M)
	PUSHDATA(serialized RedeemScript)
output script:
	HASH160
	PUSHDATA(RedeemScriptHash)
	EQUAL
```

**P2SH**第二个阶段执行

```BTCscript
2
PUSHDATA(pubkey_1)
PUSHDATA(pubkey_2)
PUSHDATA(pubkey_3)
3
CHECKMULTISIG				//用于检验多重签名
```

Proof of Burn：在交易的output中加入return语句，无论input怎么设计，执行到return时，交易就会失败，这是用于证明销毁BTC的一种方法，也可以用于往区块链中写一些东西（比如将知识产权的东西的hash值写到return语句的后面）

前面提到的往Coinbase中写数据的方法只有获得记账权的节点才能使用，这个方法所有的节点或用户都可以使用

## BTC分叉

state fork：对区块链当前的状态有分歧导致的分叉，例如攻击导致的forking attack和协议改变导致的protocol fork

hard fork & soft fork：软分叉短暂存在，硬分叉一直存在

可能出现软分叉的情况：给某些目前协议中没有规定的域增加一些新的含义，例如coinbase域被赋予意义（比如有人建议coinbase作为UTXO集合的根哈希值）

# 以太坊

## ETH-GHOST

ETH十几秒就出一次矿，但是bitcoin和Ethereum都是运行在应用层的共识协议，底层依赖P2P协议进行传播，区块发布之后传到网络上需要十几秒的时间，当两个矿工几乎同时挖到矿的时候区块链上面就会产生临时性的分叉，这种分叉在Ethereum上面就比较常见。Ethereum中产生分支之后如果没有构成最长合法链的那个区块直接作废就对个体矿工十分不公平(正常情况下我们认为矿工得到的收益应该与拥有的算力成比例，如果个体矿工和矿池同时挖到矿形成分叉，个体矿工通常竞争不过矿池，矿池算力更强，在区块中占领的位置较好其他矿工会先收到矿池挖的区块，这样会造成恶行循环，也叫centralization bias)

GHOST协议：挖到矿，但是没有加入到最长合法链中，形成了主链上的分支区块，Ethereum也给一点出块奖励，这样的分支区块被称为是当前最长合法区块上最新加入的区块的叔父区块（uncle block）。当前区块加入到最长合法区块链时可以将自己的uncle block加入进来，这样uncle block可以得到7/8的出块奖励，自身可以得到额外1/32的出块奖励，一个区块最多可以包含两个uncle block，uncle block的哈希值包含在block header中。

GHOST协议改进：新增的区块后面又新增的区块任然可以将前面区块的uncle block加入进来，这样就解决了恶意不添加uncle block或者uncle block过多的问题。如果uncle block是自己的叔父区块，uncle block的出块奖励为原本的7/8，如果是自己父亲区块的叔父区块，则变为6/8，以此类推（7代以内，最低变为2/8），防止矿工在出块简单的区块上不停产生uncle block。对于当前区块来说，还是包含一个uncle block就得到1/32的额外出块奖励。

如果两个区块不是因为对当前状态有意见分歧，而是互相认为对面是非法区块，这样的区块没法合并。

bitcoin发布区块得到block reward和TX fee，Ethereum中得到的是block reward和gas fee，uncle block得不到gas fee。

被包含的uncle block中的交易是不被执行的，如果父亲区块的交易被执行，叔父区块的交易可能就变成了非法交易。

新发布的区块只检查uncle block挖矿难度的合法性，不检查uncle block中交易的合法性。

分叉的区块只有第一个可以被包含进去，uncle block的直系后辈区块不能被包含进主链，否则分叉攻击就变得便宜了。

## ETH-挖矿算法(ethash)

block chain is secured by mining

ASIC resistance：增加对内存的访问需求（memory hard mining puzzle）

LiteCoin：使用的puzzle基于Scrypt

Ethereum：使用16M的cache和根据cache生成出来的1G的dataset，每隔30000个区块，用于生成cache的seed会被更新。cache和dataset的初始大小为16M和1G，每隔30000个块，重新生成时就会增加初始大小的1/128，即128K和8M。

### 生成cache

通过seed节点经过一些运算得到第一个元素，然后依次取哈希，将数组从前往后填入伪随机数，从而生成16M的cache

### 生成dataset

生成dataset中的第i个元素：根据伪随机的顺序读取cache中的256个数，每次读取的数值是经过上一个数经过计算得到，初始的哈希值通过cache中的第(i%cache_size)个元素生成。假设get_int_from_item函数可以从当前算出来的哈希值计算出下一个要访问的cache元素的下标，make_item函数使用cache中前面计算出相应位置的数和当前的哈希值计算出下一个哈希值，迭代256轮就可以得到一个64字节的哈希值作为大数据集中的第i个元素。不停重复直到生成大数据集中的所有元素。

![](pictures\20250304194224.png)

### 矿工挖矿的过程

矿工先得到四个参数，header，nonce， full_size(大数据集的大小),  dataset(前面所生成的大数据集)。

根据块头的信息和nonce得到一个初始的哈希值。经过64次循环，每次循环读取大数据集中相邻的两个数，读取的位置是由当前哈希值计算出来，然后根据这个位置上的数值更新当前的哈希值，最后返回的哈希值与target进行比较。矿工挖矿过程简化如下：

![挖矿过程的伪代码](pictures\20250304192521.png)

因为计算相邻两个位置中后面一个的哈希值需要用到前面位置的哈希值，所以哈希值的计算不可省略。

挖矿的过程就是不断更新nonce的值计算哈希值，判断是否符合target的要求

### 轻节点验证的过程

轻节点不挖矿，当收到矿工发过来的一个区块时，验证函数会得到四个参数，header(区块的块头)，nonce(矿工挖出矿的区块的nonce，包含在块头中)， full_size(大数据集的数据个数)，cache。验证的过程也是64个循环，不同之处在于轻节点中没有保存dataset的数据，每次使用dataset中的数据经计算哈希值时，需要计算得到dataset相应位置的数据。

![轻节点验证的伪代码](pictures\20250304192826.png)

矿工挖矿过程中需要验证非常多的nonce，每次都从16M的cache中生的话效率就太低了

以太坊中验证一个区块的合法性计算量比bitcoin要大很多，但是可以接受

Ethereum没有出现ASIC的原因：1.ASIC resistance 2.计划从PoW转向PoS(目前还是PoW)

Pre-mining：Ethereum在创建时预留一些以太币给以太坊的开发者

Pre-sale：把Pre-mining中预留的那些币通过出售的方式换取一些资产用于加密货币的开发工作，看好这个加密货币就可以在Pre-sale的时候进行购买。

## ETH-难度调整

$$
D(H) \equiv\left\{\begin{array}{ll}
D_{0} & \text { if } H_{\mathrm{i}}=0 \\
\max \left(D_{0}, P(H)_{H_{\mathrm{d}}}+x \times \varsigma_{2}\right)+\epsilon & \text { otherwise }
\end{array}\right.
 where:
 D_{0} \equiv 131072
$$

$D(H)$表示本区块的难度，由基础部分 $ P(H)_{H_{\mathrm{d}}}+x \times \varsigma_{2}$和难度炸弹 $ \epsilon$ 部分组成

$ P(H)_{H_{\mathrm{d}}}$为父区块的难度，每个区块都是在父区块难度的基础上进行调整。

$ x \times \varsigma_{2}$ 用于自适应调整出块难度，维持稳定的出块速度。

基础部分有下界，$ D_{0} = 131072$ 。
$$
x \equiv\left\lfloor\frac{P(H)_{H_{\mathrm{d}}}}{2048}\right\rfloor
$$

$$
\quad \varsigma_{2} \equiv \max \left(y-\left\lfloor\frac{H_{\mathrm{s}}-P(H)_{H_{\mathrm{s}}}}{9}\right\rfloor,-99\right)
$$

$x$是调整的单位，$\varsigma_{2}$为调整的系数

$y$ 与父区块的uncle数有关。如果父区块包括了uncle，则$y=2$, 否则$y=1$。父块包含uncle时难度会大一个单位，因为包含uncle区块时新发行的货币量大，需要适当提高难度以保持货币发行量稳定。

难度降低的上界设置为-99。

$H_{s}$ 是本区块的时间戳，$P(H)_{H_{s}}$ 是夫区块的时间戳，均以秒(s)为单位，并规定$H_{s} > P(H)_{H_{s}}$ 。该区域保证了出块时间短则调大难度，出块时间长则调小难度

分母为9的解释（前提：维持出块时间为15秒左右，以$y=1$为例)：

​		出块时间在[1, 8]之间，$\lfloor\frac{H_{\mathrm{s}}-P(H)_{H_{\mathrm{s}}}}{9}\rfloor = 0$ 出块时间过短，难度增加一个单位。

​		出块时间在[9, 17]之间，$\lfloor\frac{H_{\mathrm{s}}-P(H)_{H_{\mathrm{s}}}}{9}\rfloor = 1$ 出块时间可以接受，难度保持不变。

​		出块时间在[18, 26]之间，出块时间过长，难度调小一个单位。


$$
\epsilon \equiv \left\lfloor 2 ^{\left\lfloor H'_i \div 100000 \right\rfloor - 2} \right\rfloor
$$

$$
H'_i \equiv \max(H_i - 3000000, 0)
$$

$H_{i}$ 表示当前的区块号

为什么要设置难度炸弹$\epsilon$ ?

​		在块数还比较小的时候，难度的调整主要是由出块时间决定的，后面难度炸弹对难度的影响才会增大。

​		降低迁移到PoS协议时发生fork的风险：挖矿难度变大，所以矿工愿意迁移。

参数说明：

​		$\epsilon$ 是2的指数函数，每十万个块就会扩大一倍，后期增长非常快，这也是难度"炸弹"的由来

​		$H'_i$ 称为fake block number，之前$\epsilon$ 是由$H_i$直接得到，实际上低估了PoS协议的开发难度，需要延长大概一年半的时间。

## 权益证明

Proof of Stake：类似股份制公司按照股份进行投票

Infanticide：在一个加密货币扼杀在发行初期

Proof of Work的缺陷：费电，费资源，加密货币的总价值比股市少得多，容易受到外部力量的影响

Proof of Deposit：每次使用一部分的货币用于降低挖矿难度，但是用于降低挖矿难度的那些货币在后续一段时间内会被锁定

Nothing at Stake(权益证明早期遇到的问题)：如果使用Proof of Deposit，当出现两个分支时可以使用自己的货币两头押注，一头押注的货币只会记录在前面一个区块之中，在另一条链上并不会被锁定

Casper the Friendly Finality Gadget(FFG)：Ethereum准备使用的权益证明

Casper引入如Validator（验证者）的概念，想要成为Validator需要缴纳一定数量的以太币作为保证金，Validator的作用在于推动系统达成共识，投票决定哪条链是最长合法链

在PoW和Casper混用的时候还是有人进行挖矿，每挖出100个区块就作为一个epoch，epoch**是否能成为一个Finality需要进行投票**，Casper规定在two-phase commit的两轮投票中都需要得到2/3以上的验证者才能通过（按照保证金的金额大小来算的）

two-phase commit：1、Prepare Message 2、Commit Message

实际系统中不再区分两个Message，而且epoch中的区块减少到50个，这一轮的投票对于上一个epoch来说是Commit Message， 对下一个epoch来说是Prepare Message。

验证者参与投票的过程也有奖励，当然有不良行为时也会有相应的处罚

验证者不投票扣除部分保证金，乱作为（给两个有冲突的分叉投票）就会被没收（销毁）所有保证金

每个验证者有一定的任期，在等待期内，其他的节点可以检举揭发这个验证者是否有不良行为，验证期过后可以取回保证金和应得的奖励

包含在Finality中的交易是否绝对不会被推翻？

​		1.仅有矿工是无法推翻的，算力多强也不行，因为这是验证者经过投票选举出来的

​		2.有1/3以上的验证者两头押注可能被推翻

Ethereum的设想是逐步从PoW过渡到PoS，挖矿得到的奖励越来越小，权益证明得到的奖励越来越多

最初不使用PoS是因为当时权益证明不成熟

## ETH-智能合约

**<u>只有合约账户才有以下定义的合约以及里面的函数</u>**

转账时转账金额value可以是0，gas fee是给发布区块的矿工的，如果为0可能交易不会被打包

每个区块发布能够消耗的汽油是有上限的，上线存在block header中的`GasLimit` 中

每个矿工在发布一个区块的时候可以在上一个区块的`GasLimit` 的基础上上调或者下调1/1024，这样可以得到所有矿工普遍认为的合理值

智能合约在运行的过程中都是在修改本地的数据结构，只有在智能合约执行完了而且区块发布之后，本地的修改才对外部可见，然后才变成区块链上的共识，如果自己的区块没能发布到区块链上面，那么本区块上交易的执行都将作废

solidity不支持多线程，以太坊这种状态机的状态必须是完全确定的，多个核对内存的访问顺序不同的话，执行的结果可能是不同的

除了多线程，其他可能导致执行结果不确定的操作也是不支持的

### 定义智能合约

智能合约是运行在区块链上的一段代码，代码的逻辑定义了合约的内容

智能合约的账户保存了合约当前的运行状态

```solidity
pragma solodity ^0.4.21                       // 声明使用solidity的版本

contract SimpleAuction{
	address public beneficiary;              // 拍卖收益人
	uint public auctionEnd;                  // 结束时间
	address public highestbidder;			 // 当前最高出价人
	mappint(address => uint) bids;			 // 所有竞拍者的出价
	address[] bidders;					     // 所有竞拍者
	
	// 需要记录的事件
	event HighestBidIncreased(address bidder, uint amount);
	event Pay2Beneficiary(address winner, uint amount);
	
	/// 以受益者地址 `_beneficiary` 的名义
	/// 创建一个简单的拍卖，拍卖时间为 `_biddingTime` 秒
	constructor(uint _biddingTime, address _beneficiary) public {
		beneficiary = _beneficiary;
		auctionEnd = now + _biddingTime;
	}
	
	/// 对拍卖进行出价，随交易一起发送的ether与之前已经发送的
	/// ether的和为本次出价。
	function bid() public payable{...
	}
	
	/// 使用withdraw模式
	/// 由投标者自己取回出价，返回是否成功
	function withdraw() public returns (bool) {...
	}
	
	/// 结束拍卖，把最高的出价发送给受益人
	function pay2Beneficiary() public returns (bool) {...
	}
}
```

constract类似C++中的class类

前五行为声明的状态变量

event表示log记录

constructor是构造函数，仅在合约创建的时候调用一次

function是成员函数，可以被外部账户或者合约账户调用

function后面的payable使函数能够接受以太币（ETH）并存入合约的地址，没有payable但是转账就会抛出异常

withdraw只是将当初出价时被锁在智能合约中的ETH取回来，并没有进行转账，所以不需要payable

**智能合约通常储存在区块链上，运行在交易中**

### 合约如何调用另一个合约中的函数？

1.直接调用

```solidity
contract A{
	event LogCallFoo(string str);
	function foo(string str) returns (uint){
		emit LogCallFoo(str);
		return 123;
	}
}

contract B{
	uint ua;
	function callAFooDirectly(address addr) public{
		A a = A(addr);
		ua = a.foo("call foo directly");
	}
}
```

如果在执行a.foo()过程中抛出错误，则`callAFooDirectly`也抛出错误，本次调用全部回滚

`ua`为执行`a.foo("call foo directly")`的返回值

可以通过.gas()和.value()调整提供的gas数量或提供一些ETH

交易只能由外部账户发起，所以上面的例子中需要有一个外部账户调用合约B里面的`callAFooDirectly()`函数，然后这个函数再调用合约A中的foo函数。

2.使用address类型的call()函数

```solidity
constract C{
	function callAFooByCall(address addr) public returns (bool){
		bytes4 funcsing = bytes4(keccak256("foo(string)"));
		if (add.call(funcsig,"call foo by func call"))
			return true;
		return false;
	}
}
```

.call()里面的第一份参数被编码成4个字节，表示要调用的函数的签名

其他参数会被扩展到32个字节，表示要调用函数的参数

上例相当于`A(addr).foo("call foo by func call")`

返回true表示被调用的函数已经执行完毕，false表示引发了一个EVM异常

也可以通过.gas()和.value()调整提供的gas数量或提供一些ETH

**第一种方法调用foo函数如果出现异常会导致B也跟着一起出错，发起回滚。第二种call()调用方法如果出现异常，call()会返回false表明调用失败，但是发起调用的函数并不会发出异常 **

3.代理调用`delegatecall()`

使用方法与call()相同，只是不能使用.value()

区别在于是否切换上下文

* `call()`切换到被调用的智能合约上下文中
* `delegatecall()`只使用给定地址的代码，其他属性(存储，余额等)都取自当前合约。`delegatecall`的目的是使用存储在另外一个合约中的代码库

### fallback()函数

```solidity
function() public [payable]{
	...
}
```

匿名函数，没有参数也没有返回值

两种情况下会被调用：

* 直接向一个合约地址转账而不加任何data

* 被调用的函数不存在

如果转帐金额不是0，同样需要声明payable，否则会抛出异常

### 智能合约的创建和运行

智能合约的代码写完后，要编译成bytecode

创建合约：外部账户发起一个转账交易到0x0的地址上

* 转账金额是0，但是要支付汽油费
* 合约的代码放在data域里

智能合约运行在EVM（Ethereum Virtual Machine）上：通过加虚拟机为智能合约的运行提供一个一致的平台，EVM的寻址空间是256位，例如前面定义的`uint`就是256位的

当智能合约发布到区块链之上后，每个矿工都可以调用它，后面举的是一个拍卖的例子；任何人调用智能合约中的`bid()` 函数进行出价都需要写到区块链里

Ethereum是一个交易驱动的状态机

* 调用智能合约的交易发布到区块链上后，每个矿工都会执行这个交易，从当前的状态确定性地转移到下一个状态

### 汽油费（gas fee）

智能合约是个图灵完备模型(Turing-complete Programming Model)

全节点怎么判断对智能合约的调用是否会出现死循环？

* Halting Problem（停机问题）不可解，所以将问题推给发起交易的人。全节点收到对智能合约的调用时，首先收取调用中可能用到的最大汽油费，根据执行的情况算出花掉了多少汽油费，未花完会退回；汽油费不够会引起回滚，已经消耗掉的汽油费不退

EVM中不同指令消耗的汽油费是不一样的：简单指令便宜，复杂指令或者需要存储状态的指令很贵

### 错误处理

Ethereum中的交易具有原子性，交易既包含普通的转账交易也包含对智能合约的调用，中途出错直接回滚

智能合约中不存在自定义的try-catch结构

可以抛出错误的语句：

* assert(bool condition):如果条件不满足就抛出------用于内部错误
* require(bool condition):如果条件不满足就跑掉------用于输入或者外部组件引起的错误
* revert()终止运行并回滚状态变动

### 嵌套调用

直接调用会导致连锁式的回滚，`call()`方式不会连锁式回滚，只会调用失败返回一个false

一个合约直接向一个合约账户里转账，没有指明调用哪个函数，仍然会引起嵌套调用（fallback()函数）

### 先挖矿还是先执行区块中的交易

block header中含有状态数，交易树，收据树的根哈希值，只有模拟完区块中所有的交易之后确认hash值之后才能不断更新随机值从而进行挖矿，如果别人先将最新区块的下一个区块发布，自己还未挖出来，自己不会得到任何补偿，之前对uncle block的补偿是对已经挖到矿的矿工的补偿，没有挖到相当于陪太子读书

如果矿工不验证别人最新发到区块链的区块，这种做法蔓延开就会威胁区块链的安全，但是不验证没法更新本地三棵树的跟哈希值，后面没办法发布正确合法的区块

如果矿工相信某个全节点也可以将全节点更新完之后的数据更新到本地

### 发布到区块链上的交易是不是全部都是能够执行的

不一定，但是可以发布上去然后扣除汽油费

### 三种发送ETH的方式

* \<address>.transfer(uing256 amount)
* \<address>.send(uint256 amount) returns (bool)
* \<address>.call.value(uint256 amount)()

\<address>表示收账方的地址，transfer和send是solidity提供的address类型中的成员函数

transfer和send都是用来发送转账的，transfer转账失败之后会进行回滚，send不会

call本意是用来函数调用的，但是也可以发起转账，不会引起连锁式回滚

transfer和send调用别的合约只转过去2300个汽油（只够写一个日志），call将自己所有的汽油都转过去

### 一个简单拍卖的例子

```solidity
pragma solodity ^0.4.21                       // 声明使用solidity的版本

contract SimpleAuction{
	address public beneficiary;              // 拍卖收益人
	uint public auctionEnd;                  // 结束时间
	address public highestbidder;			 // 当前最高出价人
	mappint(address => uint) bids;			 // 所有竞拍者的出价
	address[] bidders;					     // 所有竞拍者
	bool ended;								 // 拍卖结束之后设置位true
	
	// 需要记录的事件
	event HighestBidIncreased(address bidder, uint amount);
	event Pay2Beneficiary(address winner, uint amount);
	
	/// 以受益者地址 `_beneficiary` 的名义
	/// 创建一个简单的拍卖，拍卖时间为 `_biddingTime` 秒
	constructor(uint _biddingTime, address _beneficiary) public {
		beneficiary = _beneficiary;
		auctionEnd = now + _biddingTime;
	}
	
	/// 对拍卖进行出价，随交易一起发送的ether与之前已经发送的
	/// ether的和为本次出价。
	function bid() public payable{
		// 对于能接受以太币的函数，关键字 payable时必须的。
		
		// 拍卖尚未结束
		require(now <= auctionEnd);
		// 如果出价不够高，本次出价无效，直接报错返回
		require(bids[msg.sender]+msg.value > bids[highestBidder]);
		
		// 如果此人之前未出价，则加入到竞拍者列表中
		if (!(bids[msg.sender] == uint(0))){
			bidders.push(msg.sender);
		}
		// 本次出价比当前最高价高，取代之
		highestBidder = msg.sender;
		bids[msg.sender] += msg.value;
		emit HightBidIncreased(msg.sender, bids[msg.sender]);
	}
	
	/// 结束拍卖，把最高的出价发送给受益人，
	/// 并把未中标的出价者的钱返还
	function auctionEnd() public{
		// 拍卖已截止
		require(now > auctionEnd);
		// 该函数未被调用过
		require(!ended);
		
		// 把最高的出价发送给收益人
		beneficiary.transfer(bids[hightestBidder]);
		// 对没有竞拍成功的人，将金额退回给出价的人
		for (uint i = 0; i < bidders.length; i ++){
			address bidder = bidders[i];
			if (bidder == highestBidder) continue;
			bidder.transfer(bids[bidder]);
		}
		
		ended = true;
		emit AuctionEnded(highestBidder, bids[highestBidder]);
	}
}
```

拍卖有一个受益人beneficiary，拍卖结束前每个人都可以竞拍，但是在拍卖结束前，你用于竞拍的以太币会被锁在智能合约之中，不可中途退出，但是可以加价。拍卖结束时，出价最高的人投出去的前会给受益人，其他没有竞拍成功的人可以将投出去的前取回来

第30行`bids[msg.sender]`是一个哈希表，如果之前没有出过价就为0，`msg.value`表示当前的出价

智能合约发布到区块链中就无法更改

### hack账户引发的异常

```solidity
pragma solidity ^0.4.21;

import "./SimpleAutionV1.sol";

contract hackV1{
	
	function hack_bid(address addr) payable public{
		SimpleAutionV1 sa = SimpleAutionV1(addr);
		sa.bid.value(msg.value)();
	}
}
```

外部账户调用上述的合约账户参与上面的拍卖合约，当竞拍结束退回金额的时候由于hackV1没有定义fallback()函数导致报错，合约中的`AuctionEnd()`函数无法正确执行，也就无法发送到区块链中，相当于没有执行所有的退押金操作，效果等价于将退押金的操作回滚

Code is Law：代码发到区块链中无法篡改，但是代码有bug也没办法，智能合约如果设计得不好，可能将收到的以太币永久锁起来

### 拍卖例子的第二个版本

```solidity
pragma solodity ^0.4.21                       // 声明使用solidity的版本

contract SimpleAuction{
	address public beneficiary;              // 拍卖收益人
	uint public auctionEnd;                  // 结束时间
	address public highestbidder;			 // 当前最高出价人
	mappint(address => uint) bids;			 // 所有竞拍者的出价
	address[] bidders;					     // 所有竞拍者
	
	// 需要记录的事件
	event HighestBidIncreased(address bidder, uint amount);
	event Pay2Beneficiary(address winner, uint amount);
	
	/// 以受益者地址 `_beneficiary` 的名义
	/// 创建一个简单的拍卖，拍卖时间为 `_biddingTime` 秒
	constructor(uint _biddingTime, address _beneficiary) public {
		beneficiary = _beneficiary;
		auctionEnd = now + _biddingTime;
	}
	
	/// 对拍卖进行出价，随交易一起发送的ether与之前已经发送的
	/// ether的和为本次出价。
	function bid() public payable{
		/ 对于能接受以太币的函数，关键字 payable时必须的。
		
		// 拍卖尚未结束
		require(now <= auctionEnd);
		// 如果出价不够高，本次出价无效，直接报错返回
		require(bids[msg.sender]+msg.value > bids[highestBidder]);
		
		// 如果此人之前未出价，则加入到竞拍者列表中
		if (!(bids[msg.sender] == uint(0))){
			bidders.push(msg.sender);
		}
		// 本次出价比当前最高价高，取代之
		highestBidder = msg.sender;
		bids[msg.sender] += msg.value;
		emit HightBidIncreased(msg.sender, bids[msg.sender]);
	}
	
	/// 使用withdraw模式
	/// 由投标者自己取回出价，返回是否成功
	function withdraw() public returns (bool) {
		// 拍卖已截至
		require(now > auctionEnd);
		// 竞拍成功的人需要把钱给收益人，不可取回出价
		require(msg.sender != highestBidder);
		// 当地址有钱可取
		require(bids[msg.sender] > 0);
		
		uint amount = bids[msg.sender];
		if (msg.sender.call.value(amount)()) {
			bids[msg.sender] = 0;
			return true;
		}
		return false;
	}
	
	/// 结束拍卖，把最高的出价发送给受益人
	function pay2Beneficiary() public returns (bool) {
		//拍卖已经截至
		require(now > auctionEnd);
		// 有钱可以支付
		require(bids[highestBidder] > 0);
		
		uint amount = bids[highestBidder];
		bids[highestBidder] = 0;
		emit Pay2Beneficiary(highestBidder, bids[highestBidder]);
		
		if (!beneficiary.call.value(amount)()) {
			bids[highestBidder] = amount;
			return false;
		}
		return true;
	}
}
```

这个版本不使用循环返回押金，未拍卖成功的账户**自己调用withdraw函数取回自己的押金**

### 重入攻击(Re-entrancy Attack)

当合约账户收到ETH但未调用函数时，会立刻执行`fallback()`函数

通过`addr.send()` `addr.transfer()` `addr.call.vallue()()`三种方式付钱都会触发`addr`里的`fallback()`函数。

`fallback()` 函数由用户自己编写

```
pragma solidity ^0.4.21;

import "./SimpleAuctionV2.sol";

contract HackV2 {
	uint stack = 0;
	
	function hack_bid(address addr) payable public {
		SimpleAuctionV2 sa = SimpleAuctionV2(addr);
		sa.bid.value(msg.value)();
	}
	
	function hack_withdraw(address addr) public payable{
		SimpleAuctionV2(addr).withdraw();
	}
	
	function() public payable{
		stack += 2;
		if(msg.sender.balance >= msg.value && msg.gas > 6000 && stack < 500){
			SimpleAuctionV2(msg.sender).withdraw(); // 将退押金的操作多次进行
		}
	}
}
```

多次递归调用`withdraw()` 函数，而后面的清零操作`bids[msg.sender] = 0` 在出递归之前得不到执行

第19行是hack停止递归的三个条件：余额不足，汽油费不足，调用栈溢出

应对方法：先清零，转账不成功再恢复

修改后如下：

```solidity
    /// 使用withdraw模式
    /// 由投标者自己取回出价，返回是否成功
    function withdraw() public returns (bool) {
        // 拍卖已截至
        require(now > auctionEnd);
        // 竞拍成功的人需要把钱给收益人，不可取回出价
        require(msg.sender != highestBidder);
        // 当地址有钱可取
        require(bids[msg.sender] > 0);

        uint amount = bids[msg.sender];
        bids[msg.sender] = 0;
        if (msg.sender.send(amount) {
            bids[msg.sender] = amount;
            return true;
        }
        return false;
    }
```

经典模式：先判断条件，改变条件，再与其他的合约进行交互

转账的时候使用`send()`  转账时候发过去的汽油费只有2300个单位，不足以让接收的合约发起新的调用，只够写一个log

区块链上任何未知的合约都可能是恶意的，可能反过来调用自己并且修改状态

## ETH-TheDAO

DAO(Decentralized Autonomous Organization):去中心化的自治组织

区块链上，DAO组织的规章制度写在代码里，由区块链的共识协议来维护规章制度的正常执行

TheDAO：众筹投资基金，投入以太币换取代币；投资的项目由大家投票决定，代币越多，投票权重越大，有了收益也按照智能合约中的规章制度分配。被hacker利用先转账后清零的漏洞转走当时价值50M美元的以太币

补救措施

* 如果从hacker攻击的交易前面那一个区块前面一个区块开始分叉，那么hacker后面的那些区块包含的交易也会被回滚，要回滚必须要精确定位，只能是hacker盗取的那个交易回滚，所以方法不可行
* 首先锁定hacker的账户，然后设法将盗取的以太币退回：以太坊团队制定一个软件升级，增加规则（凡是和TheDAO相关的账户不允许做任何交易），大多数以太坊的矿工升级了软件，再此之后，新矿工挖出的区块，旧矿工认可，但是旧矿工挖出的区块新矿工可能不认可（包含与TheDAO相关的账户），但是遇到一个bug，如果判断是跟TheDAO相关的账户，不予认可，没有设置收取汽油费，从此区块链上收到大量的spam transaction attack(交易中包含与TheDAO相关的账户，浪费系统的资源)，很多矿工选择了回滚到原来的版本。软分叉的解救方法失败。
* 硬分叉的解决方法：通过软件按升级的方法，在第192万个区块产生之后强行将投入TheDAO之中的以太币转入到一个新的智能合约中，这个合约只有退钱的功能，不需要签名。没有升级的矿工继续在旧链上面挖。

新链继续称为ETH，旧链称为ETC(Ethereum Classic)

因为新旧链上的交易可能在另一个链上面合法，所以给两条链赋予了chainID

## 反思

Is smart contract really smart? anything but smart

Irrevocability is a double edged sword

Nothing is irrevocable

Is solidity the right programing language?

What does decentralization mean?

decentralized $\neq$ distributed

state machine

## Beauty Chain(美链)

  美链是一个部署在以太坊上的智能合约，有自己在代币BEC

* 没有自己的区块链，代币的发行、转账都是通过调用智能合约中的函数来完成的
* 可以自己定义发行规则，每个账户有多少代币也是保存在智能合约的状态变量里
* ERC 20是以太坊上发行代币的一个标准，规范了所有发行代币的合约应该实现的功能和遵循的接口
* 美链中有一个叫`batchTransfer`的函数，它的功能是向多个接收者发送代币，然后把这些代币从调用者的账户上扣除

```solidity
function batchTransfer(address[] _receivers, uint256 _value) public whenNotPaused returns (bool) {
	uint cnt = _receivers.length;
	uint256 amount = uint256(cnt) * _value;
	require(cnt > 0 && balances[msg.sender] >= amount);
	
	balances[msg.sender] = balances[msg.sender].sub(amount);
	for (uint i = 0; i < cnt; i ++){
		balances[_receivers[i]] = balance[_receivers[i]].add(_value);
		Transfer(msg.sender, _receivers[i], _value);
	}
	return true;
}
```

代码第3行中amount可能会发生溢出，这样第6行每个账户就减少了比较少的代币，在第8行中每个接收者收到了很多数量的代币，结果是凭空出现了很多代币

**预防措施**

SafeMath库：只要通过SafeMath提供的乘法计算amount，就可以很容易地检测到溢出，第6和第8行就用到了