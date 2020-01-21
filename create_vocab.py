import re
import jieba
import pandas as pd

train_data_path = 'data/AutoMaster_TrainSet.csv'
test_data_path = 'data/AutoMaster_TestSet.csv'

stop_word_path='data/stopwords/哈工大停用词表.txt'

def load_dataset(train_data_path, test_data_path):
    '''
    数据数据集
    :param train_data_path:训练集路径
    :param test_data_path: 测试集路径
    :return:
    '''
    # 读取数据集
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    return train_data, test_data


train_df,test_df= load_dataset(train_data_path, test_data_path)
print('train data size {},test data size {}'.format(len(train_df),len(test_df)))
train_df.head()

train_df.describe()

train_df.info()

# train_df.dropna(subset=['Question', 'Dialogue', 'Report'], how='any', inplace=True)
# test_df.dropna(subset=['Question', 'Dialogue'], how='any', inplace=True)

train_df = train_df.fillna('')
test_df = test_df.fillna('')

def clean_sentence(sentence):
    '''
    特殊符号去除
    :param sentence: 待处理的字符串
    :return: 过滤特殊字符后的字符串
    '''
    if isinstance(sentence, str):
        return re.sub(
            r'[\s+\-\|\!\/\[\]\{\}_,$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好',
            '', sentence)
    else:
        return ''

sentence='2012款奔驰c180怎么样，维修保养，动力，值得拥有吗'
print('orign sentence :{} \n'.format(sentence))
print('clean sentence: {} \n'.format(clean_sentence(sentence)))

sentence='2010款的宝马X1，2011年出厂，2.0排量'
print(list(jieba.cut(sentence)))

user_dict='data/user_dict.txt'

# 载入自定义词典
jieba.load_userdict(user_dict)
print(list(jieba.cut(sentence)))

def load_stop_words(stop_word_path):
    '''
    加载停用词
    :param stop_word_path:停用词路径
    :return: 停用词表 list
    '''
    # 打开文件
    file = open(stop_word_path, 'r', encoding='utf-8')
    # 读取所有行
    stop_words = file.readlines()
    # 去除每一个停用词前后 空格 换行符
    stop_words = [stop_word.strip() for stop_word in stop_words]
    return stop_words

# 输入停用词路径 读取停用词
stop_words=load_stop_words(stop_word_path)
print('stop words size {}'.format(len(stop_words)))

# 过滤停用词
def filter_stopwords(words):
    '''
    过滤停用词
    :param seg_list: 切好词的列表 [word1 ,word2 .......]
    :return: 过滤后的停用词
    '''
    return [word for word in words if word not in stop_words]

print('orign sentence :{} \n'.format(sentence))
words = list(jieba.cut(sentence))
print('words: {} \n'.format(words))
print('filter stop word : {} '.format(filter_stopwords(words)))

def sentence_proc(sentence):
    '''
    预处理模块
    :param sentence:待处理字符串
    :return: 处理后的字符串
    '''
    # 清除无用词
    sentence = clean_sentence(sentence)
    # 切词，默认精确模式，全模式cut参数cut_all=True
    words = jieba.cut(sentence)
    # 过滤停用词
    words = filter_stopwords(words)
    # 拼接成一个字符串,按空格分隔
    return ' '.join(words)

sentence_proc(sentence)

train_df.head()

def data_frame_proc(df):
    '''
    数据集批量处理方法
    :param df: 数据集
    :return:处理好的数据集
    '''
    # 批量预处理 训练集和测试集
    for col_name in ['Question', 'Dialogue']:
        df[col_name] = df[col_name].apply(sentence_proc)

    if 'Report' in df.columns:
        # 训练集 Report 预处理
        df['Report'] = df['Report'].apply(sentence_proc)
    return df

# %%time
# train_df = data_frame_proc(train_df)
# test_df = data_frame_proc(test_df)

import numpy as np
from multiprocessing import cpu_count, Pool

# cpu 数量
cores = cpu_count()
# 分块个数
partitions = cores 
 
def parallelize(df, func):
    """
    多核并行处理模块
    :param df: DataFrame数据
    :param func: 预处理函数 
    :return: 处理后的数据
    """
    # 数据切分
    data_split = np.array_split(df, partitions)
    # 进程池
    pool = Pool(cores)
    # 数据分发 合并
    data = pd.concat(pool.map(func, data_split))
    # 关闭进程池
    pool.close()
    # 执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    pool.join()
    return data

# %%time
# 多进程处理预处理
train_df = parallelize(train_df, data_frame_proc)
test_df = parallelize(test_df, data_frame_proc)

train_df.head()

# 保存数据
train_df.to_csv('data/train_seg_data.csv',index=None,header=True)
test_df.to_csv('data/test_seg_data.csv',index=None,header=True)

merged_df=pd.concat([train_df['Question'],train_df['Dialogue'],train_df['Report'],test_df['Question'],test_df['Dialogue']],axis=0)
merged_df.head()

print('train data size {},test data size {},merged_df data size {}'.format(len(train_df),len(test_df),len(merged_df)))

merged_df.to_csv('data/merged_train_test_seg_data.csv',index=None,header=False)
merged_df.head()

# %%time
# 1. 拼接combine的所有行,形成一个超大字符串
# 2. 然后按空格切开,形成全量数据的words列表
# 3. set去重
vocab=set(' '.join(merged_df).split(' '))

# 

# for key, value in enumerate(vocab):
#     vocab_file.write(value + ' ' + str(key))
#     vocab_file.write('/n')

# %%time
# 词列表
words=[]
for sentence in merged_df:
    # 合并两个list
    words+=sentence.split(' ')
# word去重
vocab=set(words)

print('sentence size:{} ,vocab :{}'.format(len(merged_df),len(vocab)))


def word_count(file_name):
    import collections
    word_freq = collections.defaultdict(int)
    with open(file_name) as f:
        for l in f:
            for w in l.strip().split():  
                word_freq[w] += 1
    return word_freq

def _word_count(file_name):
    import collections
    word_freq = collections.defaultdict(int)
    with open(file_name) as f:
        for l in f:
            for w in l.strip().split():  
                word_freq[w] += 1
    return word_freq

def build_dict(file_name, min_word_freq=10):
    word_freq = word_count(file_name) 
    print(len(word_freq))
    word_freq = filter(lambda x: x[1] > min_word_freq, word_freq.items()) # filter将词频数量低于指定值的单词删除。
    word_freq_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0]))
    # key用于指定排序的元素，因为sorted默认使用list中每个item的第一个元素从小到
    #大排列，所以这里通过lambda进行前后元素调序，并对词频去相反数，从而将词频最大的排列在最前面
    words, _ = list(zip(*word_freq_sorted))
    word_idx = dict(zip(words, range(len(words))))
    word_idx['<unk>'] = len(words) #unk表示unknown，未知单词
    return word_idx

vocab_path = "vocab.txt"
vocab_file = open(vocab_path, 'w')

vocab_dict = build_dict('data/merged_train_test_seg_data.csv', min_word_freq=40)
for key in vocab_dict.keys():
    vocab_file.write(key + ' ' + str(vocab_dict[key]))
    vocab_file.write('\n')
