import torch
import os
import random
import numpy as np
from typing import List
from link_predict import link_prediction_eval

from openke.module.model import TransE
from openke.data import TrainDataLoader, TestDataLoader

def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 设置True后，就失去了加速的效果，可能导致程序运行变慢

def load_pretrained_node2vec_without_idmap(filename, base_emb_dim, num_total_nodes):
    """
    [适用于RETA中FB15k等数据集的处理]

    loads embeddings from node2vec style file, where each line is
    node_id node_embedding
    returns tensor containing node_embeddings
    for graph nodes 0 to n-1
    (其中的node_id即为最终的 标识不同结点的 唯一id)

    emd中可能会有部分结点没有预训练的embedding特征，需要填补为0
    """
    node_embeddings = dict()
    with open(filename, "r") as f:
        #header = f.readline()
        #emb_dim = int(header.strip().split()[1])
        for line in f:
            arr = line.strip().split()
            vocab_id = int(arr[0])  # entity在字典中对应的id
            node_emb = [float(x) for x in arr[1:]]
            node_embeddings[vocab_id] = torch.tensor(node_emb)
            # print(torch.tensor(node_emb).size())

    # 填补nove2vec中缺少的node embedding，表示为空
    embedding_tensor = torch.empty(num_total_nodes, base_emb_dim)
    print("check", embedding_tensor.size(), len(node_embeddings))
    for i in range(num_total_nodes):
        # except is updated for KGAT format since some nodes are not used in the graph
        # there is no pre-trained node2vec ebmeddings for them
        try:
            embedding_tensor[i] = node_embeddings[i]
        except KeyError:  # FIXME - replaced bare except
            pass

    out = torch.tensor(embedding_tensor)
    out = embedding_tensor
    print("node2vec tensor", out.size())

    return out

def get_name_id_map_from_txt(path):
    """
    从name2id.txt文件中读取，并封装为字典返回
    :param path:
    :return:
    """
    name2id_file = open(path, "r")

    name2id_map = {}
    for line in name2id_file:
        splitted_line = line.strip().split("\t")
        name = splitted_line[0]
        id = splitted_line[1]
        name2id_map[name] = id
    name2id_file.close()

    return name2id_map

def get_test_edges(paths: List[str], sep: str):
    # edges = set()
    edges = []
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                tokens = line.strip().split(sep)
                etype = tokens[0]
                source = tokens[1]
                destination = tokens[2]
                label = tokens[3]  # TODO:这个label是干嘛的？
                edge = (etype, source, destination, label)
                # edges.add(edge)
                edges.append(edge)

    return edges


if __name__ == "__main__":

    """
    初始化参数
    """
    gpu_id = 3
    torch.cuda.set_device(gpu_id)
    # 设置随机数种子
    setup_seed(20)  # 20
    model_name = "transe"

    """
    路径相关参数
    """
    data_path = "../../benchmarks"
    data_name = "FB15k-237"  # FB15k-RETA  humans_wikidata
    # data_path = f"{data_path}/{data_name}"
    # # absolute path to pretrained embeddings
    # pretrained_embeddings = "/data/pengmiao/workplace/pycharm/SLiCE/data/{}/entity_vec.transe".format(data_name)
    # base_embedding_dim = 200
    # ent_name2id = get_name_id_map_from_txt(f"{data_path}/entityname2id.txt")
    # num_total_nodes = len(ent_name2id)

    # # 加载保存好的embedding
    # pretrained_node_embedding = load_pretrained_node2vec_without_idmap(
    #     pretrained_embeddings, base_embedding_dim, num_total_nodes)

    """
    加载模型
    """
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path="../../benchmarks/{}/".format(data_name),
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0)

    # dataloader for test
    test_dataloader = TestDataLoader("../../benchmarks/{}/".format(data_name), "link",
                                     type_constrain=False)  # sampling_mode = 'link'
    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),  # entTotal
        rel_tot=train_dataloader.get_rel_tot(),  # relTotal
        dim=200,  # Embedding维度为200
        p_norm=1,
        norm_flag=True)
    transe.load_checkpoint('../../checkpoint/{}/{}/transe.ckpt'.format("FB15K237", model_name))

    """
    Evaluation
    """
    valid_path = data_path + f"/{data_name}/valid.txt"
    valid_edges_paths = [valid_path]
    valid_edges = list(get_test_edges(valid_edges_paths, " "))
    test_path = data_path + f"/{data_name}/test.txt"
    test_edges_paths = [test_path]
    test_edges = list(get_test_edges(test_edges_paths, " "))
    print("No. edges in test data: ", len(test_edges))

    # if there is pretrained node embeddings
    link_prediction_eval(valid_edges, test_edges, transe)