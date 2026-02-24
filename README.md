# Large-Amount-Credit-Enhancement-Competing-Products-Multi-Label-Classification

### 背景：

全自动大额增信竞品多标签分类，输入一张待分类图像，若该图像属于已知类别，输出对应类别标签；若不属于已知类别，输出”other“。

## 整体思路：

结合 Visual Retrieval(视觉检索) 与OCR Re-ranking(文本重排序)，与 FAISS 向量检索构建的非参数化场景分类系统。无需训练模型，通过构建可动态更新的向量知识库，实现对新场景的实时分类与扩展。

第一步：扫描整个数据集目录，生成视觉特征向量和文本特征向量并存入 FAISS 索引。

第二步：输入一张测试图片，系统检索视觉特征最相似的 Top-K 图像。

此时需要先使用clip系列模型，先利用视觉相似度匹配topk的图像，然后对此topk个图像利用ocr模型进行文字提取，计算输入测试图片的ocr提取内容的匹配度，最后加权融合: 
$$
alpha * Visual + (1 - alpha) * Text
$$
alpha：默认为0.6，为权重系数可调整。

最终得到top1的图像的标签作为预测输出标签。

### 若先不考虑other类别的处理

| 文件结果路径 |                                      |
| ------------ | ------------------------------------ |
| clip模型     | google/siglip2-so400m-patch16-naflex |
| ocr模型      | RapidOCR                             |
| 向量维度     | 1152                                 |
| TOP_K:       | 5                                    |
|              | 测试集无other类别                    |

#### 有ocr模型：

混淆矩阵如下：

![image-20260116163115668](C:\Users\chenrui5-jk\AppData\Roaming\Typora\typora-user-images\image-20260116163115668.png)

metrics_per_class详情如下：

| Class                            | Precision | Recall | F1-Score | Support | TP   | FP   | FN   | TN   |
| -------------------------------- | --------- | ------ | -------- | ------- | ---- | ---- | ---- | ---- |
| ayh_amount                       | 1         | 0.96   | 0.9796   | 100     | 96   | 0    | 4    | 2007 |
| ayh_user_info                    | 0.98      | 0.9899 | 0.9849   | 99      | 98   | 2    | 1    | 2006 |
| dxm_amount                       | 1         | 1      | 1        | 75      | 75   | 0    | 0    | 2032 |
| dxm_user_info                    | 0.9794    | 1      | 0.9896   | 95      | 95   | 2    | 0    | 2010 |
| dxm_yqh                          | 0.9911    | 0.9737 | 0.9823   | 114     | 111  | 1    | 3    | 1992 |
| dyfxj_amount                     | 0.98      | 1      | 0.9899   | 98      | 98   | 2    | 0    | 2007 |
| dyfxj_user_info                  | 0.9897    | 0.9897 | 0.9897   | 97      | 96   | 1    | 1    | 2009 |
| jb_main_page                     | 0.949     | 1      | 0.9738   | 93      | 93   | 5    | 0    | 2009 |
| jdjr_amount                      | 0.9701    | 1      | 0.9848   | 65      | 65   | 2    | 0    | 2040 |
| jdjr_total_amount                | 0.9811    | 0.963  | 0.972    | 108     | 104  | 2    | 4    | 1997 |
| jdjr_user_info                   | 0.8772    | 1      | 0.9346   | 100     | 100  | 14   | 0    | 1993 |
| mtshf_amount                     | 1         | 0.9897 | 0.9948   | 97      | 96   | 0    | 1    | 2010 |
| mtshf_total_amount               | 1         | 0.9787 | 0.9892   | 94      | 92   | 0    | 2    | 2013 |
| mtshf_user_info                  | 0.9434    | 1      | 0.9709   | 100     | 100  | 6    | 0    | 2001 |
| ppd_amount                       | 0.98      | 1      | 0.9899   | 98      | 98   | 2    | 0    | 2007 |
| ppd_user_info                    | 1         | 1      | 1        | 99      | 99   | 0    | 0    | 2008 |
| wld_amount                       | 1         | 0.9897 | 0.9948   | 97      | 96   | 0    | 1    | 2010 |
| wld_user_info                    | 0.9877    | 0.8163 | 0.8939   | 98      | 80   | 1    | 18   | 2008 |
| zfb_my_homepage-zfb_my_name_card | 0.9896    | 0.95   | 0.9694   | 100     | 95   | 1    | 5    | 2006 |
| zsyh_amount                      | 0.9875    | 0.9875 | 0.9875   | 80      | 79   | 1    | 1    | 2026 |
| zsyh_amount_detail               | 1         | 1      | 1        | 100     | 100  | 0    | 0    | 2007 |
| zsyh_setting                     | 1         | 0.99   | 0.995    | 100     | 99   | 0    | 1    | 2007 |

| 平均精确率 (Micro Precision): 0.9801 |      |
| ------------------------------------ | ---- |
| 平均召回率 (Micro Recall): 0.9801    |      |
| 平均 F1 (Micro F1): 0.9801           |      |

#### 无ocr模型：

| 文件结果路径 |                                      |
| ------------ | ------------------------------------ |
| clip模型     | google/siglip2-so400m-patch16-naflex |
| 无ocr模型    |                                      |
| 向量维度     | 1152                                 |
| TOP_K:       | 5                                    |
|              | 测试集无other类别                    |

混淆矩阵如下：

![image-20260116163624939](C:\Users\chenrui5-jk\AppData\Roaming\Typora\typora-user-images\image-20260116163624939.png)

metrics_per_class详情如下：

| Class                            | Precision | Recall | F1-Score | Support | TP   | FP   | FN   | TN   |
| -------------------------------- | --------- | ------ | -------- | ------- | ---- | ---- | ---- | ---- |
| ayh_amount                       | 0.8879    | 0.95   | 0.9179   | 100     | 95   | 12   | 5    | 1995 |
| ayh_user_info                    | 0.8667    | 0.9192 | 0.8922   | 99      | 91   | 14   | 8    | 1994 |
| dxm_amount                       | 0.9359    | 0.9733 | 0.9542   | 75      | 73   | 5    | 2    | 2027 |
| dxm_user_info                    | 0.9247    | 0.9053 | 0.9149   | 95      | 86   | 7    | 9    | 2005 |
| dxm_yqh                          | 0.9149    | 0.7544 | 0.8269   | 114     | 86   | 8    | 28   | 1985 |
| dyfxj_amount                     | 0.7597    | 1      | 0.8634   | 98      | 98   | 31   | 0    | 1978 |
| dyfxj_user_info                  | 0.8333    | 0.9278 | 0.878    | 97      | 90   | 18   | 7    | 1992 |
| jb_main_page                     | 0.9029    | 1      | 0.949    | 93      | 93   | 10   | 0    | 2004 |
| jdjr_amount                      | 0.9455    | 0.8    | 0.8667   | 65      | 52   | 3    | 13   | 2039 |
| jdjr_total_amount                | 0.8182    | 0.8333 | 0.8257   | 108     | 90   | 20   | 18   | 1979 |
| jdjr_user_info                   | 0.9024    | 0.74   | 0.8132   | 100     | 74   | 8    | 26   | 1999 |
| mtshf_amount                     | 1         | 0.9588 | 0.9789   | 97      | 93   | 0    | 4    | 2010 |
| mtshf_total_amount               | 0.9462    | 0.9362 | 0.9412   | 94      | 88   | 5    | 6    | 2008 |
| mtshf_user_info                  | 0.9434    | 1      | 0.9709   | 100     | 100  | 6    | 0    | 2001 |
| ppd_amount                       | 0.9655    | 0.8571 | 0.9081   | 98      | 84   | 3    | 14   | 2006 |
| ppd_user_info                    | 0.8348    | 0.9697 | 0.8972   | 99      | 96   | 19   | 3    | 1989 |
| wld_amount                       | 0.9896    | 0.9794 | 0.9845   | 97      | 95   | 1    | 2    | 2009 |
| wld_user_info                    | 0.9       | 0.8265 | 0.8617   | 98      | 81   | 9    | 17   | 2000 |
| zfb_my_homepage-zfb_my_name_card | 0.9412    | 0.8    | 0.8649   | 100     | 80   | 5    | 20   | 2002 |
| zsyh_amount                      | 0.8916    | 0.925  | 0.908    | 80      | 74   | 9    | 6    | 2018 |
| zsyh_amount_detail               | 1         | 1      | 1        | 100     | 100  | 0    | 0    | 2007 |
| zsyh_setting                     | 0.9895    | 0.94   | 0.9641   | 100     | 94   | 1    | 6    | 2006 |

| 平均精确率 (Micro Precision): 0.9079 |      |
| ------------------------------------ | ---- |
| 平均召回率 (Micro Recall): 0.9079    |      |
| 平均 F1 (Micro F1): 0.9079           |      |

#### 平均精确率 (Micro Precision)

1. 全局汇总 ：首先统计所有类别（包括 "Unknown"）的 TP (真阳性) 、 FP (假阳性) 和 FN (假阴性) 的总和。

   - $TP_{total} = \sum_{i=1}^{n} TP_i$ （所有类别预测正确的总数）
   - $FP_{total} = \sum_{i=1}^{n} FP_i$ （所有类别预测错误的“误报”总数）
   - $FN_{total} = \sum_{i=1}^{n} FN_i$ （所有类别预测错误的“漏报”总数）
2. 计算指标 ：基于这些全局总和来计算 Precision、Recall 和 F1。

### 2. 数学公式

- Micro Precision (微平均精确率) $$ Precision_{micro} = \frac{TP_{total}}{TP_{total} + FP_{total}} $$ 含义：在所有被预测出的标签中，有多少是对的。
- Micro Recall (微平均召回率) $$ Recall_{micro} = \frac{TP_{total}}{TP_{total} + FN_{total}} $$ 含义：在所有真实的标签样本中，有多少被正确预测了。
- Micro F1-Score (微平均 F1 分数) $$ F1_{micro} = 2 \times \frac{Precision_{micro} \times Recall_{micro}}{Precision_{micro} + Recall_{micro}} $$ 含义：精确率和召回率的调和平均数。

本项目作为 单标签多分类 (Single-label Multi-class Classification) 问题（即每个样本只有一个真实标签，也只预测一个标签）：

- $Precision_{micro} = Recall_{micro} = F1_{micro} = Accuracy$
- 因为每一个错误的预测，对于真实类别来说是 FN，对于预测类别来说是 FP。因此全局来看 $FP_{total} = FN_{total}$，且分母 $TP + FP$ 等于总样本数。

## 出现的问题：

### 1.with ocr模型的bad cases分析：

<img src="C:\Users\chenrui5-jk\Desktop\RAG分类\RAG\wld_user_info\个人信息.jpg" alt="个人信息" style="zoom: 25%;" /><img src="C:\Users\chenrui5-jk\Desktop\RAG分类\RAG\dxm_user_info\个人信息.jpg" alt="个人信息" style="zoom: 25%;" />

#### 分析：

首先对于clip系列的视觉模型来说，视觉内容极度相似 ： wld_user_info 、 jdjr_user_info 、 mtshf_user_info 等的“个人信息”页面在视觉布局上极其相似（都是白色背景、列表式布局）。

其次对于ocr文本内容提取来说，此系列user_info个人信息页内提取的文字内容相似度也是极高的，普遍的APP在个人信息页都会设置（姓名，性别，手机号，身份证号码，住址类信息，区分度很低），这引起了一个问题即使进行了视觉内容和ocr文字内容的融合评价，也无法精准预测这种类型的标签。

## 2.other类别

当前想实现的是一个完全的自动化的流程，其中包括当输入为other类别的待检测图像的话，模型可以自动的判断分为other类，目前遇到的有两个问题：

第一，使用图片相似度阈值法直接过滤。，我们目的是想要找到一个明显的threshold在已知的类的版式与所有未知的图像版式之间，按照常理来分析，如果是未知的other类的待预测图片送入图片相似度检测与已知类别的图片送入相似度检测，显然总体的相似度平均值会更低一些，如果能有这样的一个threshold存在，那么我们把这些低于这个threshold的待分类图片可以全部分入other类别即可，但是可惜的是，经过对目前的数据集，已知的类别与other类别进行相似度分析之后，并没有看到明显的这样的一个threshold，经过初步分析，原因是不论是否是已知的类别还是other类别的版式界面，除了其中的文字内容，版式布局可能存在不同之外，绝大多数的背景区域是非常相似的，而使用的clip系列模型并没有如此强的细粒度感知的能力，这源于CLIP在预训练中侧重于粗粒度语义匹配，缺乏对局部判别性特征的关注，也就导致了CLIP的对比学习范式使其更关注**全局语义对齐**，而**忽视局部细节特征**，由此影响，我们发现，相似度最低的待预测图像与数据库中存入的图像向量的相似度也超过了0.8，这使得我们如果使用clip系列作为图像编码器编码图像特征向量的话，几乎不可能找到这样的一个threshold。

<img src="C:\Users\chenrui5-jk\AppData\Roaming\Typora\typora-user-images\image-20260119143230290.png" alt="image-20260119143230290" style="zoom:50%;" />

第二，我们还发现，目前测试集属于other类别的某些版式与已知类别的版式非常的接近，严重怀疑这些竞品的版式是模仿一些主流公司的版式进行设计的，仅仅是布局，色调的些许差距，以及关键文本的不同，如以下两张other类别的版式与dyfxj_amount此类非常相似，并且这两张other版式内部仅仅是“天津金城银行”和“亿联银行”的文字差异，其余完全一致，这对于我们不论是使用视觉特征向量，亦或者是使用ocr相似度检测来说，都是极难的样例，目前没有特别好的全自动化的分类方法。

<img src="C:\Users\chenrui5-jk\AppData\Roaming\Typora\typora-user-images\image-20260119145218101.png" alt="image-20260119145218101" style="zoom:20%;" /><img src="C:\Users\chenrui5-jk\AppData\Roaming\Typora\typora-user-images\image-20260119145228763.png" alt="image-20260119145228763" style="zoom:20%;" /><img src="C:\Users\chenrui5-jk\AppData\Roaming\Typora\typora-user-images\image-20260119150004324.png" alt="image-20260119150004324" style="zoom:25%;" />



### 下一步研究：

基于上述clip系列模型的不足，我们尝试更换模型去提取图片特征向量，按照需求，具有细粒度感知能力的模型肯定会提升我们分类的精准度，并且也可能解决我们other类别无法判断的问题，我们由此思路入手，展开进一步的研究与测试。

#### 通义千问3-VL-Embedding-8B
