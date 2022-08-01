import argparse
from data_loader import load_data
from train import train
import time
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
begin=time.time()

#parser解析器
parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, default='./data/news/train.txt', help='path to the training file')
parser.add_argument('--test_file', type=str, default='./data/news/test.txt', help='path to the test file')
parser.add_argument('--transform', type=bool, default=True, help='whether to transform entity embeddings')#是否转换实体嵌入
parser.add_argument('--use_context', type=bool, default=False, help='whether to use context embeddings')#是否使用上下文嵌入
parser.add_argument('--max_click_history', type=int, default=30, help='number of sampled click history for each user')#每位用户点击历史的数目
parser.add_argument('--n_filters', type=int, default=128, help='number of filters for each size in KCNN')#kcnn滤波器数目
parser.add_argument('--filter_sizes', type=int, default=[1, 2], nargs='+',
                    help='list of filter sizes, e.g., --filter_sizes 2 3')#滤波器数目
parser.add_argument('--l2_weight', type=float, default=0.01, help='weight of l2 regularization')#l2正则化权重
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')#学习速率
parser.add_argument('--batch_size', type=int, default=128, help='number of samples in one batch')#每批batch的大小
parser.add_argument('--n_epochs', type=int, default=100,help='number of training epochs')#训练次数
parser.add_argument('--KGE', type=str, default='TransE',#KGE的创建方式
                    help='knowledge graph embedding method, please ensure that the specified input file exists')#知识图嵌入方法，请确保指定的输入文件存在
parser.add_argument('--entity_dim', type=int, default=50,
                    help='dimension of entity embeddings, please ensure that the specified input file exists')#实体嵌入的尺寸，请确保指定的输入文件存在
parser.add_argument('--word_dim', type=int, default=50,
                    help='dimension of word embeddings, please ensure that the specified input file exists')#word嵌入的尺寸，请确保指定的输入文件存在
parser.add_argument('--max_title_length', type=int, default=10,
                    help='maximum length of news titles, should be in accordance with the input datasets')#新闻标题的最大长度，应与输入的数据集一致

args = parser.parse_args()

train_data, test_data = load_data(args)
train(args, train_data, test_data)


end=time.time()
print("训练用时：",int(end-begin),"s")
