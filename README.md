# DuReader数据集
DuReader是一个新的大型真实世界和人类来源的MRC数据集。 DuReader专注于真实的开放域问题解答。 DuReader优于现有数据集的优点总结如下：

- 真正的问题
- 真实的文章
- 真实答案
- 实际应用场景
- 丰富的注释
# DuReader基线系统
DuReader系统在DuReader数据集上实现了2个经典阅读理解模型（BiDAF和Match-LSTM）。 该系统使用2个框架实现：PaddlePaddle和TensorFlow。
## 如何运行
### 下载数据集
要下载DuReader数据集：


```
cd data && bash download.sh
```

有关DuReader数据集的更多详细信息，请参阅DuReader主页。

### 下载Thirdparty Dependencies
我们使用Bleu和Rouge作为评估指标，这些指标的计算依赖于“https://github.com/tylin/coco-caption”
下的评分脚本，下载它们，运行：


```
cd utils && bash download_thirdparty.sh
```


### 预处理数据
下载数据集后，仍有一些工作要做，以运行基线系统。 DuReader数据集为每个用户问题提供了大量文档，文档对于流行的RC模型来说太长了。在我们的基线模型中，我们通过选择与答案字符串最相关的段落来预处理列车集和开发集数据，而对于推断（没有可用的黄金答案），我们选择与问题字符串最相关的段落。预处理策略在utils / preprocess.py中实现。要预处理原始数据，首先应分段'问题'，'标题'，'段落'，然后将分段结果存储到'segmented_question'，'segmented_title'，'segmented_paragraphs'中，如下载的预处理数据，然后运行：


```
cat data / raw / trainset / search.train.json | python utils / preprocess.py> data / preprocessed / trainset / search.train.json
```

预处理数据可以通过data / download.sh自动下载，并存储在数据/预处理中，预处理前的原始数据在data / raw之下。

### 运行PaddlePaddle
我们使用PaddlePaddle实现BiDAF模型。请注意，我们在PaddlePaddle基线上有更新（2019年2月25日）。 paddle / UPDATES.md中已注明主要更新。在DuReader的数据集上，PaddlePaddle基线比我们的Tensorflow基线具有更好的性能。 PaddlePaddle基线也支持多gpu训练。

PaddlePaddle基准包括以下程序：段落提取，词汇准备，培训，评估和推理。所有这些过程都包含在paddle / run.sh中。您可以通过运行带有特定参数的run.sh来启动一个过程。基本用法是：


```
sh run.sh --PROCESS_NAME --OTHER_ARGS
```

PROCESS_NAME可以是para_extraction，准备，培训，评估和预测之一（请参阅以下每个程序的详细说明）。 OTHER_ARGS是特定的参数，可以在paddle / args.py中找到。

在下面的示例中（“段落提取”除外），我们默认使用demo数据集（在data / demo下）来演示paddle / run.sh的用法。

### 环境要求
请注意，我们仅在PaddlePaddle v1.2（Fluid）上测试了基线。要安装PaddlePaddle，请参阅PaddlePaddle主页以获取更多信息。

### 段落提取
我们采用了一种新的段落提取策略来改善模型性能。在paddle / UPDATES.md中已经注明了细节。请运行以下命令以在每个文档上应用段提取的新策略：


```
sh run.sh --para_extraction
```

请注意，在运行此命令之前，应准备好完整的预处理数据集（请参阅上面的“预处理数据”部分）。段落提取的结果将保存在数据/提取/中。只有在运行完整数据集时才需要此过程，如果您只想尝试使用演示数据进行词汇准备/培训/评估/推理，则可以使用此算法。

### 词汇准备
在训练模型之前，您需要为数据集准备词汇表并创建将用于存储模型和结果的文件夹。您可以运行以下命令进行准备：


```
sh run.sh --prepare
```

上面的命令默认使用data / demo /中的数据。要更改数据文件夹，您需要指定以下参数：

```
sh run.sh --prepare --trainset ../data/extracted/trainset/zhidao.train.json ../data/extracted/trainset/search.train.json --devset ../data/extracted/devset/zhidao.dev.json ../data/extracted/devset/search.dev.json --testset ../data/extracted/testset/zhidao.test.json ../data/extracted/testset/search.test.json
```


### 训练
要训练模型（在演示列车上），请运行以下命令：


```
sh run.sh --train --pass_num 5
```

这将开始5个时期的训练过程。在每个纪元之后将自动评估训练的模型，并且将在文件夹数据/模型下创建由纪元ID命名的文件夹，其中保存模型参数。如果您需要更改默认的超参数，例如初始学习率和隐藏大小，请运行具有特定参数的命令。


```
sh run.sh --train --pass_num 5 --learning_rate 0.00001 --hidden_size 100
```

可以在paddle / args.py中找到更多参数。

### 评估
要评估特定模型（在演示版上），请运行以下命令：


```
sh run.sh --evaluate --load_dir YOUR_MODEL_DIR
```

将加载并评估YOUR_MODEL_DIR下的模型（例如../data/models/1）。

### 推理（预测）
要使用经过训练的模型进行推理（在演示测试集上），请运行：


```
sh run.sh --predict --load_dir YOUR_MODEL_DIR
```

预测的答案将保存在文件夹数据/结果中。

### Dudleader 2.0上PaddlePaddle Baseline的性能

型号 | Dev ROUGE-L | 测试ROUGE-L
---|---|---
更新前 | 39.29 | 45.90
更新后 | 47.65 | 54.58



上表中的结果是通过使用批量大小= 4 * 32的4个P40 GPU卡获得的。如果使用具有较小批量（例如32）的单张卡，性能可能略低，但应该高于设备上的ROUGE-L = 47。

### 提交测试结果
一旦您训练了在开发集上调整的模型，我们强烈建议您将测试集的预测提交到DuReader的站点以进行评估。要在测试集上获取推理文件：

1. 确保培训结束。
2. 根据训练日志选择数据/模型下的最佳模型。
3. 预测测试集的结果。
4. 提交预测结果文件。

## 运行Tensorflow
我们还实现了基于Tensorflow 1.0的BIDAF和Match-LSTM模型。您可以参考Tensorflow安装的官方指南。可以使用python run.py -h访问运行Tensorflow程序的完整选项。这里我们演示一个典型的工作流程如下：

### 制备
在训练模型之前，我们必须确保数据准备就绪。为了准备，我们将检查数据文件，制作目录并提取词汇表供以后使用。您可以运行以下命令以使用指定的任务名称执行此操作：


```
python run.py --prepare
```

您可以通过设置train_files / dev_files / test_files来指定train / dev / test的文件。默认情况下，我们使用data / demo /中的数据

### 训练
要训练阅读理解模型，可以使用--algo [BIDAF | MLSTM]指定模型类型，也可以使用--learning_rate NUM设置学习率等超参数。例如，要训练10个时期的BIDAF模型，您可以运行：


```
python run.py --train --algo BIDAF --epochs 10
```

培训过程包括在每个培训时期之后对开发集进行评估。默认情况下，将保存开发集上具有最少Bleu-4分数的模型。

### 评估
要使用已经训练过的模型对dev set进行单一评估，可以运行以下命令：


```
python run.py --evaluate --algo BIDAF
```

### 预测
您还可以使用以下命令预测某些文件中样本的答案：


```
python run.py --predict --algo BIDAF --test_files ../data/demo/devset/search.dev.json
```

默认情况下，结果保存在../data/results/文件夹中。 您可以通过指定--result_dir DIR_PATH来更改此设置。

## 在多语言数据集上运行基线系统
为帮助评估多语言数据集的系统性能，我们提供脚本将MS MARCO V2数据从其格式转换为DuReader格式。

MS MARCO（微软机器阅读理解）是一个专注于机器阅读理解和问答的英语数据集。 MS MARCO和DuReader的设计类似。值得在中文（DuReader）和英文（MS MARCO）数据集上检查MRC系统。

您可以下载MS MARCO V2数据，并运行以下脚本将数据从MS MARCO V2格式转换为DuReader格式。然后，您可以在MS MARCO数据上运行和评估我们的DuReader基线或DuReader系统。


```
./run_marco2dureader_preprocess.sh ../data/marco/train_v2.1.json ../data/marco/train_v2.1_dureaderformat.json
./run_marco2dureader_preprocess.sh ../data/marco/dev_v2.1.json ../data/marco/dev_v2.1_dureaderformat.json
```

## 版权和许可
版权所有2017 Baidu.com，Inc。保留所有权利

根据Apache许可证2.0版（“许可证”）获得许可;除非符合许可，否则您不得使用此文件。您可以在以下位置获取许可证副本


```
http://www.apache.org/licenses/LICENSE-2.0
```

除非适用法律要求或书面同意，否则根据许可证分发的软件将按“原样”分发，不附带任何明示或暗示的担保或条件。有关管理许可下的权限和限制的特定语言，请参阅许可证。
























