import argparse
import time

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
begin = time.time()
from dkn import DKN
import tensorflow as tf
from data.news.news_preprocess import *
from data_loader import read,aggregate,load_test

tf.reset_default_graph()

PATTERN1 = re.compile('[^A-Za-z]')#字符串处理
PATTERN2 = re.compile('[ ]{2,}')
WORD_FREQ_THRESHOLD = 2 #word频率阈值
ENTITY_FREQ_THRESHOLD = 1 #entity频率阈值
MAX_TITLE_LENGTH = 10
WORD_EMBEDDING_DIM = 50

word2freq = {}
entity2freq = {}
word2index = {}
entity2index = {}
corpus = []

#统计单词与实体的数量
def count_word_and_entity_freq(files):
    """
    Count the frequency of words and entities in news titles in the training and test files
    :param files: [training_file, test_file]
    :return: None
    """
    for file in files:
        reader = open(file, encoding='utf-8')
        for line in reader:
            array = line.strip().split('\t')
            news_title = array[1]
            entities = array[3]

            # count word frequency
            for s in news_title.split(' '):
                if s not in word2freq:
                    word2freq[s] = 1
                else:
                    word2freq[s] += 1

            # count entity frequency
            for s in entities.split(';'):
                entity_id = s[:s.index(':')]
                if entity_id not in entity2freq:
                    entity2freq[entity_id] = 1
                else:
                    entity2freq[entity_id] += 1

            corpus.append(news_title.split(' '))
        reader.close()

#构建单词id文件、实体索引文件
def construct_word2id_and_entity2id():
    """
    Allocate each valid word and entity a unique index (start from 1)
    :return: None
    """
    cnt = 1  # 0 is for dummy word
    for w, freq in word2freq.items():
        if freq >= WORD_FREQ_THRESHOLD:
            word2index[w] = cnt
            cnt += 1
    print('- word size: %d' % len(word2index))

    writer = open('data/kg/entity2index.txt', 'w', encoding='utf-8')
    cnt = 1
    for entity, freq in entity2freq.items():
        if freq >= ENTITY_FREQ_THRESHOLD:
            entity2index[entity] = cnt
            writer.write('%s\t%d\n' % (entity, cnt))  # for later use
            cnt += 1
    writer.close()
    print('- entity size: %d' % len(entity2index))


def get_local_word2entity(entities):
    """
    Given the entities information in one line of the dataset, construct a map from word to entity index
    E.g., given entities = 'id_1:Harry Potter;id_2:England', return a map = {'harry':index_of(id_1),
    'potter':index_of(id_1), 'england': index_of(id_2)}
    :param entities: entities information in one line of the dataset
    :return: a local map from word to entity index
    """
    local_map = {}

    for entity_pair in entities.split(';'):
        entity_id = entity_pair[:entity_pair.index(':')]
        entity_name = entity_pair[entity_pair.index(':') + 1:]

        # remove non-character word and transform words to lower case
        entity_name = PATTERN1.sub(' ', entity_name)
        entity_name = PATTERN2.sub(' ', entity_name).lower()

        # constructing map: word -> entity_index
        for w in entity_name.split(' '):
            entity_index = entity2index[entity_id]
            local_map[w] = entity_index

    return local_map

#新闻标题进行数字编码
def encoding_title(title, entities):
    """
    Encoding a title according to word2index map and entity2index map
    :param title: a piece of news title
    :param entities: entities contained in the news title
    :return: encodings of the title with respect to word and entity, respectively
    """
    local_map = get_local_word2entity(entities)

    array = title.split(' ')
    word_encoding = ['0'] * MAX_TITLE_LENGTH
    entity_encoding = ['0'] * MAX_TITLE_LENGTH

    point = 0
    for s in array:
        if s in word2index:
            word_encoding[point] = str(word2index[s])
            if s in local_map:
                entity_encoding[point] = str(local_map[s])
            point += 1
        if point == MAX_TITLE_LENGTH:
            break
    word_encoding = ','.join(word_encoding)
    entity_encoding = ','.join(entity_encoding)
    return word_encoding, entity_encoding

def transform_test(line):
    array = line.strip().split('\t')
    user_id = array[0]
    title = array[1]
    entities = array[2]
    word_encoding, entity_encoding = encoding_title(title, entities)
    return '%s\t%s\t%s\n' % (user_id, word_encoding, entity_encoding)

def get_word2vec_model():
    if not os.path.exists('word_embeddings_' + str(WORD_EMBEDDING_DIM) + '.model'):
        print('- training word2vec model...')
        w2v_model = gensim.models.Word2Vec(corpus, size=WORD_EMBEDDING_DIM, min_count=1, workers=16)
        print('- saving model ...')
        w2v_model.save('word_embeddings_' + str(WORD_EMBEDDING_DIM) + '.model')
    else:
        print('- loading model ...')
        w2v_model = gensim.models.Word2Vec.load('word_embeddings_' + str(WORD_EMBEDDING_DIM) + '.model')
    return w2v_model


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
def get_feed_dict(model, data, start, end):
    feed_dict = {model.clicked_words: data.clicked_words[start:end],
                 model.clicked_entities: data.clicked_entities[start:end],
                 model.news_words: data.news_words[start:end],
                 model.news_entities: data.news_entities[start:end]}
    return feed_dict


def test(args):
    print('counting frequencies of words and entities ...')
    count_word_and_entity_freq(['data/news/raw_train.txt', 'data/news/raw_test.txt'])

    print('constructing word2id map and entity to id map ...')
    construct_word2id_and_entity2id()
    train_df = read(args.train_file)
    uid2words, uid2entities = aggregate(train_df, args.max_click_history)
    model = DKN(args)

    with tf.Session() as sess:
        saver = tf.train.Saver()  # 定义保存的对象
        saver.restore(sess, 'model/DKNModel')
        test_data = None
        while test_data != 'a':
            test_data = input('请输入：')
            # test_data='21000	kesha new music coming soon according label	10258:Kesha'
            test_data = transform_test(test_data)
            test_data = load_test(test_data, args,uid2words, uid2entities)
            scores = model.test(sess, get_feed_dict(model, test_data, 0, 1))
            print('scores',scores)
            if scores > 0.64:
                print('点击')
            else:
                print('不点击')

def main():
    # parser解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='./data/news/train.txt', help='path to the training file')
    parser.add_argument('--test_file', type=str, default='./data/news/test.txt', help='path to the test file')
    parser.add_argument('--transform', type=bool, default=True,
                        help='whether to transform entity embeddings')  # 是否转换实体嵌入
    parser.add_argument('--use_context', type=bool, default=False,
                        help='whether to use context embeddings')  # 是否使用上下文嵌入
    parser.add_argument('--max_click_history', type=int, default=30,
                        help='number of sampled click history for each user')  # 每位用户点击历史的数目
    parser.add_argument('--n_filters', type=int, default=128,
                        help='number of filters for each size in KCNN')  # kcnn滤波器数目
    parser.add_argument('--filter_sizes', type=int, default=[1, 2], nargs='+',
                        help='list of filter sizes, e.g., --filter_sizes 2 3')  # 滤波器数目
    parser.add_argument('--l2_weight', type=float, default=0.01, help='weight of l2 regularization')  # l2正则化权重
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # 学习速率
    parser.add_argument('--batch_size', type=int, default=128, help='number of samples in one batch')  # 每批batch的大小
    parser.add_argument('--n_epochs', type=int, default=100, help='number of training epochs')  # 训练次数
    parser.add_argument('--KGE', type=str, default='TransE',  # KGE的创建方式
                        help='knowledge graph embedding method, please ensure that the specified input file exists')  # 知识图嵌入方法，请确保指定的输入文件存在
    parser.add_argument('--entity_dim', type=int, default=50,
                        help='dimension of entity embeddings, please ensure that the specified input file exists')  # 实体嵌入的尺寸，请确保指定的输入文件存在
    parser.add_argument('--word_dim', type=int, default=50,
                        help='dimension of word embeddings, please ensure that the specified input file exists')  # word嵌入的尺寸，请确保指定的输入文件存在
    parser.add_argument('--max_title_length', type=int, default=10,
                        help='maximum length of news titles, should be in accordance with the input datasets')  # 新闻标题的最大长度，应与输入的数据集一致
    parser.add_argument('--val_txt', type=str, default='./val.txt',
                        help='val path')  # 新闻标题的最大长度，应与输入的数据集一致

    args = parser.parse_args()
    test(args)


if __name__ == '__main__':
    main()
