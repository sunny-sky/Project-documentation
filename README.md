# DuReader���ݼ�
DuReader��һ���µĴ�����ʵ�����������Դ��MRC���ݼ��� DuReaderרע����ʵ�Ŀ����������� DuReader�����������ݼ����ŵ��ܽ����£�

- ����������
- ��ʵ������
- ��ʵ��
- ʵ��Ӧ�ó���
- �ḻ��ע��
# DuReader����ϵͳ
DuReaderϵͳ��DuReader���ݼ���ʵ����2�������Ķ����ģ�ͣ�BiDAF��Match-LSTM���� ��ϵͳʹ��2�����ʵ�֣�PaddlePaddle��TensorFlow��
## �������
### �������ݼ�
Ҫ����DuReader���ݼ���


```
cd data && bash download.sh
```

�й�DuReader���ݼ��ĸ�����ϸ��Ϣ�������DuReader��ҳ��

### ����Thirdparty Dependencies
����ʹ��Bleu��Rouge��Ϊ����ָ�꣬��Щָ��ļ��������ڡ�https://github.com/tylin/coco-caption��
�µ����ֽű����������ǣ����У�


```
cd utils && bash download_thirdparty.sh
```


### Ԥ��������
�������ݼ�������һЩ����Ҫ���������л���ϵͳ�� DuReader���ݼ�Ϊÿ���û������ṩ�˴����ĵ����ĵ��������е�RCģ����˵̫���ˡ������ǵĻ���ģ���У�����ͨ��ѡ������ַ�������صĶ�����Ԥ�����г����Ϳ��������ݣ��������ƶϣ�û�п��õĻƽ�𰸣�������ѡ���������ַ�������صĶ��䡣Ԥ���������utils / preprocess.py��ʵ�֡�ҪԤ����ԭʼ���ݣ�����Ӧ�ֶ�'����'��'����'��'����'��Ȼ�󽫷ֶν���洢��'segmented_question'��'segmented_title'��'segmented_paragraphs'�У������ص�Ԥ�������ݣ�Ȼ�����У�


```
cat data / raw / trainset / search.train.json | python utils / preprocess.py> data / preprocessed / trainset / search.train.json
```

Ԥ�������ݿ���ͨ��data / download.sh�Զ����أ����洢������/Ԥ�����У�Ԥ����ǰ��ԭʼ������data / raw֮�¡�

### ����PaddlePaddle
����ʹ��PaddlePaddleʵ��BiDAFģ�͡���ע�⣬������PaddlePaddle�������и��£�2019��2��25�գ��� paddle / UPDATES.md����ע����Ҫ���¡���DuReader�����ݼ��ϣ�PaddlePaddle���߱����ǵ�Tensorflow���߾��и��õ����ܡ� PaddlePaddle����Ҳ֧�ֶ�gpuѵ����

PaddlePaddle��׼�������³��򣺶�����ȡ���ʻ�׼������ѵ������������������Щ���̶�������paddle / run.sh�С�������ͨ�����д����ض�������run.sh������һ�����̡������÷��ǣ�


```
sh run.sh --PROCESS_NAME --OTHER_ARGS
```

PROCESS_NAME������para_extraction��׼������ѵ��������Ԥ��֮һ�����������ÿ���������ϸ˵������ OTHER_ARGS���ض��Ĳ�����������paddle / args.py���ҵ���

�������ʾ���У���������ȡ�����⣩������Ĭ��ʹ��demo���ݼ�����data / demo�£�����ʾpaddle / run.sh���÷���

### ����Ҫ��
��ע�⣬���ǽ���PaddlePaddle v1.2��Fluid���ϲ����˻��ߡ�Ҫ��װPaddlePaddle�������PaddlePaddle��ҳ�Ի�ȡ������Ϣ��

### ������ȡ
���ǲ�����һ���µĶ�����ȡ����������ģ�����ܡ���paddle / UPDATES.md���Ѿ�ע����ϸ�ڡ�������������������ÿ���ĵ���Ӧ�ö���ȡ���²��ԣ�


```
sh run.sh --para_extraction
```

��ע�⣬�����д�����֮ǰ��Ӧ׼����������Ԥ�������ݼ������������ġ�Ԥ�������ݡ����֣���������ȡ�Ľ��������������/��ȡ/�С�ֻ���������������ݼ�ʱ����Ҫ�˹��̣������ֻ�볢��ʹ����ʾ���ݽ��дʻ�׼��/��ѵ/����/���������ʹ�ô��㷨��

### �ʻ�׼��
��ѵ��ģ��֮ǰ������ҪΪ���ݼ�׼���ʻ�����������ڴ洢ģ�ͺͽ�����ļ��С����������������������׼����


```
sh run.sh --prepare
```

���������Ĭ��ʹ��data / demo /�е����ݡ�Ҫ���������ļ��У�����Ҫָ�����²�����

```
sh run.sh --prepare --trainset ../data/extracted/trainset/zhidao.train.json ../data/extracted/trainset/search.train.json --devset ../data/extracted/devset/zhidao.dev.json ../data/extracted/devset/search.dev.json --testset ../data/extracted/testset/zhidao.test.json ../data/extracted/testset/search.test.json
```


### ѵ��
Ҫѵ��ģ�ͣ�����ʾ�г��ϣ����������������


```
sh run.sh --train --pass_num 5
```

�⽫��ʼ5��ʱ�ڵ�ѵ�����̡���ÿ����Ԫ֮���Զ�����ѵ����ģ�ͣ����ҽ����ļ�������/ģ���´����ɼ�ԪID�������ļ��У����б���ģ�Ͳ������������Ҫ����Ĭ�ϵĳ������������ʼѧϰ�ʺ����ش�С�������о����ض����������


```
sh run.sh --train --pass_num 5 --learning_rate 0.00001 --hidden_size 100
```

������paddle / args.py���ҵ����������

### ����
Ҫ�����ض�ģ�ͣ�����ʾ���ϣ����������������


```
sh run.sh --evaluate --load_dir YOUR_MODEL_DIR
```

�����ز�����YOUR_MODEL_DIR�µ�ģ�ͣ�����../data/models/1����

### ����Ԥ�⣩
Ҫʹ�þ���ѵ����ģ�ͽ�����������ʾ���Լ��ϣ��������У�


```
sh run.sh --predict --load_dir YOUR_MODEL_DIR
```

Ԥ��Ĵ𰸽��������ļ�������/����С�

### Dudleader 2.0��PaddlePaddle Baseline������

�ͺ� | Dev ROUGE-L | ����ROUGE-L
---|---|---
����ǰ | 39.29 | 45.90
���º� | 47.65 | 54.58



�ϱ��еĽ����ͨ��ʹ��������С= 4 * 32��4��P40 GPU����õġ����ʹ�þ��н�С����������32���ĵ��ſ������ܿ����Եͣ���Ӧ�ø����豸�ϵ�ROUGE-L = 47��

### �ύ���Խ��
һ����ѵ�����ڿ������ϵ�����ģ�ͣ�����ǿ�ҽ����������Լ���Ԥ���ύ��DuReader��վ���Խ���������Ҫ�ڲ��Լ��ϻ�ȡ�����ļ���

1. ȷ����ѵ������
2. ����ѵ����־ѡ������/ģ���µ����ģ�͡�
3. Ԥ����Լ��Ľ����
4. �ύԤ�����ļ���

## ����Tensorflow
���ǻ�ʵ���˻���Tensorflow 1.0��BIDAF��Match-LSTMģ�͡������Բο�Tensorflow��װ�Ĺٷ�ָ�ϡ�����ʹ��python run.py -h��������Tensorflow���������ѡ�����������ʾһ�����͵Ĺ����������£�

### �Ʊ�
��ѵ��ģ��֮ǰ�����Ǳ���ȷ������׼��������Ϊ��׼�������ǽ���������ļ�������Ŀ¼����ȡ�ʻ���Ժ�ʹ�á���������������������ʹ��ָ������������ִ�д˲�����


```
python run.py --prepare
```

������ͨ������train_files / dev_files / test_files��ָ��train / dev / test���ļ���Ĭ������£�����ʹ��data / demo /�е�����

### ѵ��
Ҫѵ���Ķ����ģ�ͣ�����ʹ��--algo [BIDAF | MLSTM]ָ��ģ�����ͣ�Ҳ����ʹ��--learning_rate NUM����ѧϰ�ʵȳ����������磬Ҫѵ��10��ʱ�ڵ�BIDAFģ�ͣ����������У�


```
python run.py --train --algo BIDAF --epochs 10
```

��ѵ���̰�����ÿ����ѵʱ��֮��Կ���������������Ĭ������£������濪�����Ͼ�������Bleu-4������ģ�͡�

### ����
Ҫʹ���Ѿ�ѵ������ģ�Ͷ�dev set���е�һ���������������������


```
python run.py --evaluate --algo BIDAF
```

### Ԥ��
��������ʹ����������Ԥ��ĳЩ�ļ��������Ĵ𰸣�


```
python run.py --predict --algo BIDAF --test_files ../data/demo/devset/search.dev.json
```

Ĭ������£����������../data/results/�ļ����С� ������ͨ��ָ��--result_dir DIR_PATH�����Ĵ����á�

## �ڶ��������ݼ������л���ϵͳ
Ϊ�����������������ݼ���ϵͳ���ܣ������ṩ�ű���MS MARCO V2���ݴ����ʽת��ΪDuReader��ʽ��

MS MARCO��΢������Ķ���⣩��һ��רע�ڻ����Ķ������ʴ��Ӣ�����ݼ��� MS MARCO��DuReader��������ơ�ֵ�������ģ�DuReader����Ӣ�ģ�MS MARCO�����ݼ��ϼ��MRCϵͳ��

����������MS MARCO V2���ݣ����������½ű������ݴ�MS MARCO V2��ʽת��ΪDuReader��ʽ��Ȼ����������MS MARCO���������к��������ǵ�DuReader���߻�DuReaderϵͳ��


```
./run_marco2dureader_preprocess.sh ../data/marco/train_v2.1.json ../data/marco/train_v2.1_dureaderformat.json
./run_marco2dureader_preprocess.sh ../data/marco/dev_v2.1.json ../data/marco/dev_v2.1_dureaderformat.json
```

## ��Ȩ�����
��Ȩ����2017 Baidu.com��Inc����������Ȩ��

����Apache���֤2.0�棨�����֤����������;���Ƿ�����ɣ�����������ʹ�ô��ļ���������������λ�û�ȡ���֤����


```
http://www.apache.org/licenses/LICENSE-2.0
```

�������÷���Ҫ�������ͬ�⣬����������֤�ַ������������ԭ�����ַ����������κ���ʾ��ʾ�ĵ������������йع�������µ�Ȩ�޺����Ƶ��ض����ԣ���������֤��
























