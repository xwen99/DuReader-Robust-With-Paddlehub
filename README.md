# 基于~~魔改~~官方 PaddleHub Baseline 的 DuReader<sub>robust</sub> 解决方案

## 任务说明

* 本次评测的阅读理解数据集 (即DuReader robust) 旨在衡量阅读理解模型的鲁棒性，评测模型的过敏感性、过稳定性以及泛化能力。该数据集包括训练集、开发集以及测试集：
  * 训练集（领域内，in-domain）: 该训练集是利用大规模中文阅读理解数据集DuReader 2.0中的实体类问题构建而成的，共包含15K个样本；数据集中的问题均为百度搜索中的真实用户问题，数据集中的文档均来自百度搜索结果中的普通网页或百度知道中的问答页面。
  * 开发集（领域内，in-domain）：开发集的构造方法和来源与训练集相同，共包含1.4K个样本。
  * 测试集：本次评测所使用的测试集包含四个部分，一个领域内（in-domain）测试集和三个鲁棒性测试集，具体如下：
  * 领域内（in-domain）测试集: 该部分的构造方法和来源与训练集、开发集相同，共包含1.3K个样本。
    * 过敏感（over-sensitivity）测试集：该部分首先对领域内测试集中的样本进行随机采样，然后对样本中的问题生成复述，构成新的样本。新的样本用于测试模型在相同的篇章上，对于语义相同、表述不同的复述问题，是否能正确的、稳定的产生相同的回答。如果模型对这些复述问题生成了不同的、错误的答案，则我们称模型存在过敏感的问题。与之前的相关工作不同，该部分生成的复述问题，都是搜索引擎中用户问过的真实问题。该部分共包含1.3K个样本。
    * 过稳定（over-stability）测试集：该部分首先对领域内数据集中的样本通过规则进行采样，被选取样本的篇章中包含多个和参考答案类型相同的实体（称为干扰答案）。然后由专业人员根据篇章、参考答案以及干扰答案标注新的问题，构成新的样本。其中，新标注的问题与干扰答案所在句（称为干扰句）之间有许多相同的词汇。如果模型不能够区分答案句和干扰句，即模型过于稳定地仅通过字面进行匹配，而未进行语义匹配，则我们称模型存在过稳定的问题。与之前的相关工作不同，该部分的篇章都是真实篇章，而非通过构造、拼接干扰句构造而成。该部分共包含0.8K个样本。
    * 泛化能力（generalization）测试集：该部分样本的领域分布与训练集的领域分布不同，主要由教育领域（数学、物理、化学、语文等）以及金融领域的真实文本构建而成。该部分用于测试模型在未知领域（out-of-domain）上的泛化能力。如果模型在该部分的效果下降较多，则我们称模型的泛化能力不够好。该部分共包含1.6K个样本。

* 数据集的论文链接：[DuReader<sub>robust</sub>: A Chinese Dataset Towards Evaluating the Robustness of
Machine Reading Comprehension Models](https://arxiv.org/pdf/2004.11142.pdf)
* 数据格式：类似 SQuAD 1.x

## 探索过程与解决方案

* 写在前面
  * 个人之前只有一些 CV 方面的炼丹经历，对于阅读理解任务还是首次尝试，因此主要边学习边探索。选择 PaddleHub Baseline 一方面因为较为方便，一方面因为另一个官方 Baseline 在多卡训练和 fp16 上都遇到了 bug，而我又懒得再用 PyTorch 写个 Baseline。。。后面果不其然又遇到了不少 PaddleHub 的坑，因此不建议在此项目基础上继续开发，但方便大家理解与上手还是不错的。采用日常碎片时间做了两周多的时间，在 test1 数据集上达到了 77 的 F1-score，与 80+ 的前排大佬还有不小差距，姑且作为个人学习锻炼过程的记录吧。 
* 思路
  * 本赛题是典型的抽取式阅读理解任务，即给定 question 问题与 context 文档，要求从文档中截取片段作为答案。对于每个问题，只有一个文档，而且文档长度并不是很长，和其他相关比赛相比难度并不在答案检索方面，而是鲁棒性。这方面最直接的思路便是花式加数据，另外还有对抗训练、多任务学习等等。
  * Baseline 采用的方案是 ERNIE 提取字级别特征（中文是字，非中文序列整体视为一个字），然后对答案序列起止位置当做两个分类任务来做。具体而言，字级别特征送 fc 层输出两个 logits，然后分别计算两个 softmax loss，取平均。非常经典而简洁的方案，而且看其他人比赛经验分享普遍表示各种魔改效果都不如保持这样。。。
* 数据
  * 加数据体现在两方面：基于现有数据做数据增强，以及直接加上其他中文阅读理解数据集。
    * 数据增强方面，推荐看这篇文章：[NLP中数据增强的综述，快速的生成大量的训练数据](https://zhuanlan.zhihu.com/p/142168215)，以及苏神的这篇文章里的 trick：[基于CNN的阅读理解式问答模型：DGCNN](https://kexue.fm/archives/5409)。我有尝试过用 Google 翻译做回译（中->英->中），但不论对问题和文档都做回译，还是只对问题做回译，都只会掉点。分析原因可能是百度知道和百科的语料质量较低（翻翻数据你就明白了，这倒也不怪百度。。），导致回译后对语义影响较大。另外有论文（[Improving the Robustness of Question Answering Systems to Question Paraphrasing](https://www.aclweb.org/anthology/P19-1610.pdf)）表示对问题做转写（paraphrase）对鲁棒性会有一定提升，但由于未找到合适的中文转写方案，遂作罢。最后采用的方案是随机将问题和材料的部分字 id 置零。
    * 加数据集方面，考虑了 CMRC2018, DRCD（繁转简）, DuReader 2.0（格式转换脚本见 dureader2squad.py）, WebQA, SogouQA, CAIL2019。DRCD 和 CAIL2019 基本没提升，WebQA 和 SogouQA 甚至会拉胯，可能还是任务 domain 有一定差异。经测试最后采用了 CMRC2018 和 DuReader 2.0。最后提交的模型把开发集也加入训练，对于多答案的情况随机抽一个做 ground truth。
* 模型
  * 推荐看这个项目：[中文语言理解基准测评](https://github.com/CLUEbenchmark/CLUE)，详细比较了各中文预训练模型在各语言任务的表现。在百度的论文里目前最好的中文预训练模型应该是 ERNIE 2.0，奈何百度不开源中文的权重。在和 ERNIE 苦苦纠缠了几天之后果断转战 RoBERTa-wwm-large，虽然模型真的重，但结果真的香（直接提 3 个点）。
  * 模型其他方面感觉没有多大改进空间，预训练模型还是强，前人经验表示 fc 层一把梭就是最好的方法，简单试了一下加两层 Transformer 会掉点，就没再改。
* 训练
  * 模型训练中会出现开发集 loss 一直涨但点数也一直涨的现象，这一方面是由于评价指标（F1-score 与 EM）和损失函数的偏差，一方面由于简单负样本占多数。ACL 2020 有一篇文章（[Dice Loss for Data-imbalanced NLP Tasks](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1911.02855.pdf)）讲自己做了可微版本的 F1-score 并且结合了 Focal loss 的技巧，但亲测会掉点，等开源吧。后期把 Softmax loss 换成了 Focal loss，有一点点提升，看苏神的讲解就好了：[何恺明大神的「Focal Loss」，如何更好地理解？](https://zhuanlan.zhihu.com/p/32423092)。
  * 提升模型鲁棒性另一个思路是对抗训练。百度 AAAI 2020 有一篇文章（[A Robust Adversarial Training Approach to Machine Reading Comprehension](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-LiuK.6841.pdf)）效果很不错，但稍微有点复杂。另外推荐两篇文章：[【炼丹技巧】功守道：NLP中的对抗训练 + PyTorch实现](https://fyubang.com/2019/10/15/adversarial-train/)，[对抗训练浅谈：意义、方法和思考（附Keras实现）](https://kexue.fm/archives/7234)。这里我复现了 FGM，魔改 PaddleHub 的 Task 真的遇到一堆坑。。。效果上，结果确实变得非常鲁棒了，对于同一问题的很多变种都能保证答案的一致，但点数上并没有提很多，分析是因为之前还可以有的答对有的答错，现在要么全对要么全错了 emmm...
  * 另外一个方向就是多任务学习。百度的 [D-NET](https://www.aclweb.org/anthology/D19-5828/) 在 MRQA 2019 大比分拿了第一，论文里也讲了一个多任务学习的故事，但事实上最终提交的模型只是裸的 XL-Net + ERNIE 2.0 的 ensemble。预训练模型强就是可以为所欲为啊。。。另外文中提到不同模型的 ensemble 要比同模型的 ensemble 效果好不少，但亲测只有两个模型的单模型效果都不错才可以。另外调研过程中意外发现百度的开源多任务学习框架 [PALM](https://github.com/PaddlePaddle/PALM) 竟然是小瑶姐开发的。虽然实际没搞多任务学习，还是要强烈推荐下她的公众号「夕小瑶的卖萌屋」，真的又可爱又涨知识。
  * 其他训练参数方面，没变，只要调就掉点，两个 epoch 足矣。
* 评测
  * PaddleHub 的 Baseline 默认用的 CMRC2018 的评测代码，这里手工换成了 LIC2020（即本次比赛）的，数值上有一定偏差。
  * max_answer_length 默认就好，40 也可以，调低对开发集有一点提示但测试集会变差。答案生成步骤实际用了答案起止两个位置 logit 评分的和对 nbest 答案进行排序，对此评分准侧做过一些修改的尝试但没有提升。最后对答案长度加了一点惩罚，即上述评分 - lambda * (end - start)。
  * 最后提交的结果为 5 个 roberta 的 ensemble
* 结果
  * test1 数据集 F1 76.98849，排名 37
  * test2 数据集 F1 72.887，排名 15

## 踩坑记录
* 可能需要 export cuda 路径到 LD_LIBRARY_PATH
* paddle.fluid. gradients 不支持求跨 program 的梯度，加 program_guard 也无解。
* paddle.fluid.backward.append_backward 支持求跨 program 的梯度，但 clone program 的时候，如果是 test 模式，会被删掉，因此最好 clone 之后再 append_backward。
* ~~`adv_loss = self.cl_loss_from_embedding(perturb + self.feature)` 这句话，paddle 1.7 以前有 bug，1.7 以后必须按这个顺序，交换后（`self.feature + perturb`）也有 bug。~~ （paddle 2.0.0 已修复）
* ~~PaddleHub 保存模型 checkpoint 在 Paddle 1.7 及以后会有 bug，解决方案是把整个 io 文件替换成旧的。~~（paddle 1.8.x 已修复）

## 代码使用
* 先 `bash download.sh` 下载数据
* 然后 `bash train.sh` 训练模型
* 然后 `bash evaluate.sh` 测试结果（如有必要）
* 然后 `bash predict.sh` 跑 test 集的预测
* 最后 `python ensemble.py` 做模型 ensemble（如有需要）
* 参数要求见程序及脚本示例

## 前人经验
* [第四届语言与智能高峰论坛-会议资料下载](http://tcci.ccf.org.cn/summit/2019/dl.php)
* [lic2019-dureader2.0-rank2](https://github.com/SunnyMarkLiu/lic2019-dureader2.0-rank2)
* [BERT-Dureader](https://github.com/HandsomeCao/BERT-Dureader)
* [LES-MMRC-Summary](https://github.com/YingZiqiang/LES-MMRC-Summary)
* [DIAC2019-Adversarial-Attack-Share](https://github.com/WenRichard/DIAC2019-Adversarial-Attack-Share)
* [cail2019](https://github.com/NoneWait/cail2019)
* [基于CNN的阅读理解式问答模型：DGCNN](https://kexue.fm/archives/5409)
*  [【比赛分享】刷新CoQA榜单纪录：基于对抗训练和知识蒸馏的机器阅读理解方案解析](https://fyubang.com/2019/11/06/coqa/)

## 可能的改进方向
* 更好的预训练模型（如果有）
* 更多的数据增强（如问题重述）
* 更多的手工特征（如词特征、共现特征等）
* 更多的训练技巧（如多任务训练）
* 更好的对抗训练（比如复现百度 AAAI 2020 那篇文章）
* 加知识图谱
* 加知识蒸馏
* 。。。
