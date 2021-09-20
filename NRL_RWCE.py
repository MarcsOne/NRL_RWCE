#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from gensim.models import Word2Vec
from collections import defaultdict
import random
import networkx as nx
import copy
import time
import math
import evaluator

# 加载网络
def load_graph(path):
    G = nx.Graph()
    with open(path) as text:
        for line in text:
            vertices = line.strip().split("\t")
            source = int(vertices[0])
            target = int(vertices[1])
            if source!=target:
                G.add_edge(source, target)
                G.add_edge(target, source)
            else:
                G.add_node(source)

    return G
def init1(G,old_dict,new_dict):
    first_del = defaultdict(bool)
    for node in G.nodes():
        dict = defaultdict(float)
        dict[node] = 1
        old_dict[node] = dict
        new_dict[node] = defaultdict(float)
        first_del[node] = False
    cores = Rough_Core(G)
    for index,group in enumerate(cores):
        com_lable = group[0]
        for node in group:
            if node in old_dict[node].keys() and first_del[node] == False:
                del old_dict[node][node]
                first_del[node] = True
            old_dict[node][com_lable] = 1
    # 归一化
    for i in range(G.number_of_nodes()):
        normalize(old_dict[i])

def init2(G,old_dict,new_dict):
    for node in G.nodes():
        old_dict[node] = defaultdict(float)
        new_dict[node] = defaultdict(float)

    cores = Rough_Core(G)
    for index,group in enumerate(cores):
        com_lable = index
        for node in group:
            old_dict[node][com_lable] = 1.0
    for node in G.nodes():
        if len(old_dict[node]) == 0:
            old_dict[node][-node] = 1.0
    # 归一化
    for node in G.nodes():
        normalize(old_dict[node])


def normalize(lable_dict):
    sumb = 0
    for lable , b in lable_dict.items():
        sumb += b
    for lable in lable_dict.keys():
        if sumb != 0:
            lable_dict[lable] = lable_dict[lable] / sumb
def id_x(x):
    ids = []
    for item in x.keys():
        ids.append(item)
    return ids
# 统计这一次迭代后形成的社区的编号集合 ids
def id_l(l):
    ids = set()
    for node in l.keys():
        ids = ids.union(set(id_x(l[node])))
    return ids
# 统计此刻网络中每个社区标签关联的结点数
def count(l):
    counts = defaultdict(float)
    for node in l.keys():
        for lable in l[node].keys():
            if lable in counts.keys():
                n = counts[lable]
                counts[lable] = n + 1
            else:
                counts[lable] = 1
    return counts

# 社区关联结点数取小
def mc(cs1,cs2):
    cs = {}
    for each in cs1:
        if each in cs2:
            cs[each] = min(cs1[each],cs2[each])
    return cs

# 一次标签同步更新过程
def propagate(node,old_dict,new_dict,p,node_nei):
    # 邻居传播顺序随机
    # random.shuffle(node_nei)
    # 根据邻居标签集更新节点node标签集
    # print(node_nei)
    # print(old_dict[0])
    new_dict[node].clear()
    # print(new_dict[node])
    flag=True
    # print("node_nei",node_nei)
    for nei in node_nei:
        # flag = False
        for lable in new_dict[nei].keys():
            # flag = False
            b = new_dict[nei][lable] # 得到节点的一个社区归属度
            # 如果节点node标签集包含这个标签,则隶属度累加,否则整个标签进行传播
            if lable in new_dict[node]:
                # flag=False
                new_dict[node][lable] += b
            else:
                # flag = False
                new_dict[node][lable] = b
    # print(new_dict[node])
    # print(flag)
    # 归一化节点标签隶属度
    # print("before:\n",new_dict[node])
    normalize(new_dict[node])
    # print("after:\n",new_dict[node])
    temp_pop = []
    # 寻找bmax
    bmax = 0
    lable_max = 0
    for lable,b in new_dict[node].items():
        if b > bmax:
            bmax = b
            lable_max = lable

    # 去除节点标签集中b/bmax小于p的节点标签
    for lable in new_dict[node].keys():
        # print(float(new_dict[node][lable]/bmax))
        if float(new_dict[node][lable]/bmax) < float(p):
            temp_pop.append(lable)
    for i in temp_pop:
        new_dict[node].pop(i)
    normalize(new_dict[node])

    # print("delete: ",new_dict[node])

def Rough_Core(G):
    # 获取节点的度，字典
    degrees_dict = nx.degree(G)
    # print(type(degrees_dict))
    # 按度从大到小排序节点，得到vd_l ist
    vd_list = sorted(degrees_dict.items(), key=lambda x: x[1], reverse=True)
    # 是否已经加入某个团标志
    num_nodes = G.number_of_nodes()
    free_flags = defaultdict(bool)

    # 生成最小的最大团
    cores = []
    for elem in vd_list:
        # 度大于3且未入团
        core = []
        if elem[1] >= 3:
            nei_node = G.neighbors(elem[0])
            # 找出当前节点邻居中具有最大度的点且标记为已入团
            max_node = 0
            max_degree = 0
            for node in nei_node:
                if G.degree(node) >= max_degree:
                    max_degree = G.degree(node)
                    max_node = node
            # 入团并标记为已入团
            # free_flags[elem[0]] = True
            # free_flags[max_node] = True
            core.append(max_node)
            core.append(elem[0])
            # 求两个核心点的公共邻居
            commNeiber_set = set(G.neighbors(max_node)) & set(nei_node)
            degrees_comm = defaultdict(int)
            for node in commNeiber_set:
                degrees_comm[node] = G.degree(node)
            # 按度从小到大排序
            degrees_comm = sorted(degrees_comm.items(), key=lambda x: x[1])
            while len(commNeiber_set) != 0:
                for item in degrees_comm:
                    if item[0] in commNeiber_set:
                        # 节点h加入
                        core.append(item[0])
                        # free_flags[item[0]] = True
                        commNeiber_set = commNeiber_set & set(G.neighbors(item[0]))
                        # commNeiber_set.remove(item[0])
                # 更新 sorted degree list
                # degrees_comm.clear()
                # for node in commNeiber_set:
                #     degrees_comm.append((node,G.degree(node)))
                # degrees_comm = sorted(degrees_comm, key=lambda x: x[1])
        if len(core) >= 3:
            cores.append(core)
    return cores
# BMLPA算法过程
# p:平衡因子，越大越平衡。（0,1]
def BMLPA(G,p = 0.5,maxiter = 1000,flag=True):
    old_dict = defaultdict(dict)
    new_dict = defaultdict(dict)
    # minl = {}
    # oldmin = {}
    itera = 0
    # 初始化节点标签为各自的编号
    if flag:
        init1(G, old_dict,new_dict)
        # print(old_dict)
    else:
        init2(G, old_dict, new_dict)
        # print(old_dict)
    oldmin = count(old_dict)
    # print(old_dict)
    # print(new_dict)
    # 节点标签迭代传播(同步更新)
    new_dict = copy.deepcopy(old_dict)
    start = time.clock()
    while True:
        itera += 1
        for node in G.nodes():
            # 同步更新节点标签
            node_nei = G.neighbors(node)
            if len(node_nei) > 0:
                # print(old_dict[0])
                propagate(node, old_dict, new_dict,p,node_nei)
        # print(old_dict)
        if id_l(old_dict) == id_l(new_dict):
            # 如果t时刻与t-1时刻网络中存在的标签编号集合是一致的
            # 对比t-1时刻各社区关联结点数minl，和t时刻count（new_dict），设置此时minl中每个社区关联结点数为两者中更小的一方。
            # 即若t-1时刻，社区ID9关联7个节点，t时刻，社区ID9关联9个节点，那么认为此时其关联7个节点。
            minl = mc(oldmin, count(new_dict))
        else:
            # 如果标签编号不一致，直接社区minl为t时刻中每个社区关联的结点数集合
            minl = count(new_dict)
            # 如果该时刻计算出来的minl与上一时刻oldmin是一致的 ，即存在于网络中的每个社区编号都是一样的，
            # 且各社区关联结点数也与上一时刻一样，那么结束迭代。
            # 若不一致，则继续迭代。
        if minl != oldmin:
            # print("before:\n",old_dict)
            # print("new_dict:\n",new_dict)
            old_dict = copy.deepcopy(new_dict)
            # print("after:\n", old_dict)
            oldmin = copy.deepcopy(minl)
        elif minl == oldmin:
            break
        if itera == maxiter:
            break
    # print(old_dict)
    print("itera: ", itera)

    elapsed = (time.clock() - start)
    print("BMLPA Time cost: ", elapsed)


    coms = defaultdict(set)
    sub = defaultdict(set)
    for node in G.nodes():
        ids = set(id_x(old_dict[node]))
        for each_c in ids:
            if each_c in coms and each_c in sub:
                coms[each_c].add(node)
                sub[each_c] = ((set(sub[each_c]) & set(ids)))
            else:
                coms[each_c] = set([node])
                sub[each_c] = set(ids)
    # 利用sub去重社区
    for each in sub:
        if len(sub[each]):
            for e in sub[each]:
                if e != each:
                    coms[e] = coms[e] - coms[each]
    # 删除空白社区
    coms_blank = []
    for num_c, node_list in coms.items():
        if len(node_list) == 0:
            coms_blank.append(num_c)
    for each in coms_blank:
        coms.pop(each)

    print('Communities: ', coms)
    print("Number of community: ", len(coms))
    return new_dict,coms


def build_my_corpus(G,lable_dict,num_paths, path_length,x, rand=random.Random(0)):

    overlapping_nodes = defaultdict(list)
    c = 0
    for node in lable_dict.keys():
        if len(lable_dict[node]) > 1:
            for nei in G.neighbors(node):
                if len(lable_dict[nei]) > 1:
                    overlapping_nodes[node].append(nei)
            c += 1
    print("number of overlapping nodes: ", c)

    walks = []
    nodes = list(G.nodes())

    # 预计算转移矩阵
    tri_dict = generate_tri_dict(G,lable_dict)

    start = time.clock()

    for node in nodes:
        repeat = 0
        if G.degree(node)>num_paths:
            repeat = num_paths
        else:
            repeat = G.degree(node)
        # if len(lable_dict[node])>1:
        #     repeat=20
        for cnt in range(repeat):
            if G.degree(node) <= x:
                walks.append(overlapping_nodes_sample(G,overlapping_nodes,lable_dict,tri_dict,path_length,rand=rand,start=node))
            else:
                walks.append(random_walk_membership_based(G, lable_dict, tri_dict, path_length, rand=rand,start=node))

    rand.shuffle(walks)
    elapsed = (time.clock() - start)
    print("Random Walk Time cost: ", elapsed)
    return walks

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s



def random_walk_None_based(G,path_length,rand=random.Random(),start=None):  # 生成随机游走序列

    path = [start]

    while len(path) < path_length:
        cur = path[-1]    # 选定序列最后一个节点为当前节点
        if len(path)>=2:
            prior = path[-2] #选定当前节点的上一个节点
        else:
            prior = path[-1]
        if len(G.neighbors(cur))>0:   # 如果有边与当前节点相连
            # x = random.random()
            # if x > 0.3:
            #     next_node = np.random.choice(a=list(G.neighbors(cur)), size=1)[0]
            # else:
            #     next_node = start
            next_node = np.random.choice(a=list(G.neighbors(cur)), size=1)[0]
            path.append(next_node) # 随机选取一个作为下一个当前节点
        else:
            break
    return [str(node) for node in path] # 返回一条随机游走序列

def random_walk_membership_based(G,lable_dict,tri_dict,path_length,rand=random.Random(),start=None):  # 生成随机游走序列

    path = [start]
    # else:
    #     rand.choice(list(G.keys()))  # 如果给定起始节点则由该节点开始游走，否则随机选取起始节点
    while len(path) < path_length:
        cur = path[-1]    # 选定序列最后一个节点为当前节点
        if len(path)>=2:
            prior = path[-2] #选定当前节点的上一个节点
        else:
            prior = path[-1]
        if len(G.neighbors(cur))>0:   # 如果有边与当前节点相连
            # x = random.random()
            # if x > 0.2:
            #     next_node = np.random.choice(a=list(tri_dict[cur].keys()), size=1, p=list(tri_dict[cur].values()))[0]
            # else:
            #     next_node = start
            # average_degree = 10
            # if G.degree(cur) < average_degree:
            #     next_node = np.random.choice(a=list(G.neighbors(cur)), size=1)[0]
            # else:
            #     next_node = np.random.choice(a=list(tri_dict[cur].keys()), size=1, p=list(tri_dict[cur].values()))[0]

            next_node = np.random.choice(a=list(tri_dict[cur].keys()), size=1, p=list(tri_dict[cur].values()))[0]
            path.append(next_node) # 随机选取一个作为下一个当前节点

        else:
            break
    return [str(node) for node in path] # 返回一条随机游走序列

def random_walk_density_based(G,k,N,a=0.5):
    # 以节点度来衡量节点密度，并按节点密度从大到小排序。
    density_dict = defaultdict(float)
    # for node in G.nodes():
    #     density_dict[node] = G.degree(node)
    for node in G.nodes():
        den = 0.0
        for nei in G.neighbors(node):
            den += sigmoid(G.degree(nei))
        density_dict[node] = den
        # print(den)
    density_tuple = sorted(density_dict.items(), key=lambda x: x[1],reverse=True)
    # print(density_tuple)
    # 选取前k个节点做2k次基于密度的随机游走，两次指导策略不同，第一次随机游走过程中选择比当前节点密度小且与当前节点密度差最小的邻居转
    # 移，直到没有满足要求的下一跳节点停止，记录其游走序列长度，序列集合即为 paths_min。
    # 第二次随机游走与第一次类似，只是游走过程中选择比当前节点密度小且与当前节点密度
    # 差最大的邻居转移，直到没有满足要求的下一跳节点时停止。记录其游走序列长度，序列集合记为 paths_max。
    paths_min = []
    paths_max = []

    for i,(node,density) in enumerate(density_tuple):
        if i == k:
            break
        walks_min = [node]
        walks_max = [node]
        cur_max = node
        cur_min = node
        cur_min_density = density
        cur_max_density = density
        flag_min = True
        flag_max = True
        last_min = -1
        last_max = -1
        # visited_min = set()
        # visited_max = set()
        while(1):
            d_min = N
            d_max = 0
            node_min = -1
            node_max = -1
            # print("flag_min flag_max", flag_min, flag_max)
            # print("flag_min flag_max", cur_min, cur_max)
            if flag_min:
                for next in G.neighbors(cur_min):
                    next_density = density_dict[next]
                    # if next in visited_min:
                    #     continue
                    if next_density < cur_min_density:
                        d = cur_min_density-next_density
                        if d < d_min:
                            d_min = d
                            node_min = next

            if flag_max:
                for next in G.neighbors(cur_max):
                    next_density = density_dict[next]
                    # if next == visited_max:
                    #     continue
                    if next_density < cur_max_density:
                        d = cur_max_density-next_density
                        if d > d_max :
                            d_max = d
                            node_max = next


            if node_min == -1:
                flag_min = False
            else:
                walks_min.append(node_min)
                # visited_min.add(cur_min)
                cur_min = node_min
                cur_min_density = density_dict[cur_min]
            if node_max == -1:
                flag_max = False
            else:
                walks_max.append(node_max)
                # visited_max.add(cur_max)
                cur_max = node_max
                cur_max_density = density_dict[cur_max]


            if flag_max == False and flag_min == False:
                break
        paths_max.append(walks_max)
        paths_min.append(walks_min)
    sum_min = 0
    sum_max = 0
    print("The length of paths_min: ",len(paths_min))
    for walk in paths_min:
        sum_min += len(walk)

    print("The length of paths_max: ", len(paths_max))
    for walk in paths_max:
        sum_max += len(walk)

    pre_walk_length = sum_min/k
    pre_window_size = (1-a)*(sum_max/k) + a*pre_walk_length

    print("The prediction of walk_length: ", math.ceil(2*pre_walk_length))
    print("The prediction of window_size: ", math.ceil(pre_window_size))

    return math.ceil(2*pre_walk_length),math.ceil(pre_window_size)

# 计算两个节点node1->node2的转移概率评分（未归一化）
def cal_tri(G,node1,node2,lable_dict):
    tri = 0.0001
    # sim = sim_Salton(G.neighbors(node1),G.neighbors(node2),len(G.neighbors(node1)),len(G.neighbors(node2)))
    # t = len(lable_dict[node1]) + len(lable_dict[node2])
    count = 0
    for lable in lable_dict[node2].keys():
        if lable in lable_dict[node1].keys():
            count += 1
            tri += 1.0/(abs(lable_dict[node1][lable]-lable_dict[node2][lable]) + 1.0)
    # weight = count/(t-count+1)
    return tri

# 生成全图的转移概率表
def generate_tri_dict(G,lable_dict):
    start = time.clock()
    tri_dict = defaultdict(dict)
    for node in G.nodes():
        # 遍历每个节点，如果有邻居
        if(len(G.neighbors(node)))>0:
            tmp1_dict = dict()
            for i in G.neighbors(node):
                tri1 = cal_tri(G,node,i,lable_dict)
                tmp1_dict[i] = tri1
            # 概率归一化
            normalize(tmp1_dict)
            tri_dict[node] = tmp1_dict
    elapsed = (time.clock() - start)
    print("Precomputed transfer matrix Time cost: ", elapsed)
    return tri_dict


def aggregate_nodes(coms):
    new_nodes = defaultdict(list)
    for label, nodes in coms.items():
        if len(nodes) != 0:
            new_nodes[label] = nodes
    return new_nodes


def judge_newEdge(origin_G,nodes1,nodes2,t):
    count = 0.0
    if len(nodes1)<len(nodes2):
        nodes_min = nodes1
        nodes_max = nodes2
    else:
        nodes_min = nodes2
        nodes_max = nodes1
    for i in nodes_min:
        for j in nodes_max:
            if origin_G.has_edge(i,j):
                count += 1.0
    s = count / len(nodes_min)
    if s >= t:
        return True
    else:
        return False



def generate_newEdges(origin_G,new_nodes,t):
    new_edges = defaultdict(list)
    # number_of_newNodes = len(new_nodes.keys())
    for node1 in new_nodes.keys():
        # l = list(new_nodes.keys())[node1+1: ]
        for node2 in new_nodes.keys():
            if node1 != node2:
                if judge_newEdge(origin_G, new_nodes[node1], new_nodes[node2], t):
                    new_edges[node1].append(node2)
                    # new_edges[node2].append(node1)
    return new_edges

def generate_aggregateGraph(new_nodes,new_edges):
    G = nx.Graph()
    for node in new_nodes.keys():
        G.add_node(node)
    for node,edges in new_edges.items():
        for nei in edges:
            G.add_edge(node,nei)
    return G

# 输出到社区结果文件
def toComtxt(output_filepath,coms,):
    with open(output_filepath, 'w') as f:
        count = 0
        for elem in coms:
            for node in coms[elem]:
                f.write(" " + str(node))
            f.write("\n")
            count += 1
        # print("number of community: ", count)
def toComtxt1(output_filepath,coms,new_nodes):
    with open(output_filepath, 'w') as f:
        count = 0
        for elem in coms:
            for new_node in coms[elem]:
                for node in new_nodes[new_node]:
                    f.write(" " + str(node))
            f.write("\n")
            count += 1
        # print("number of community: ", count)

def integrate_labelDict(label_dict1,lable_dict2,new_nodes):
    for new_node,label_dict in lable_dict2.items():
        for old_node in new_nodes[new_node]:
            label_dict1[old_node].update(label_dict)
    return label_dict1


# def integrate_labelDict(lable_dict2,new_nodes):
#     lable_dict = defaultdict(dict)
#     for new_node,dic in lable_dict2.items():
#         for old_node in new_nodes[new_node]:
#             lable_dict[old_node].update(dic)
#     return lable_dict
def integrate_labelDict1(lable_dict,coms,new_nodes):
    for com_lables in coms.values():
        com_lables = list(com_lables)
        if len(com_lables)>=2:
            for com_lable in com_lables[1:]:
                for ori_node in new_nodes[com_lable]:
                    b = lable_dict[ori_node][com_lable]
                    if com_lables[0] in lable_dict[ori_node]:
                        lable_dict[ori_node][com_lables[0]] += b
                        lable_dict[ori_node].pop(com_lable)
                    else:
                        lable_dict[ori_node][com_lables[0]] = b
                        lable_dict[ori_node].pop(com_lable)
    # pass
def sim_Salton(u_nei,v_nei,len_u_nei,len_v_nei):
    sameNei = list(set(u_nei).intersection(set(v_nei)))
    num_sameNei = len(sameNei)
    sim_uv = num_sameNei / np.sqrt((len_u_nei-1) * (len_v_nei-1))
    # print(sim_uv)
    return sim_uv

def cal_average_degree(G):
    sum_degree = 0
    for node in G.nodes():
        sum_degree += G.degree(node)
    average_degree = sum_degree/nx.number_of_nodes(G)
    print("average_degree: ",average_degree)

def overlapping_nodes_sample(G,overlapping_nodes,lable_dict,tri_dict,path_length,rand=random.Random(),start=None):
    path = [start]
    f = 0
    while len(path) < path_length:
        cur = path[-1]  # 选定序列最后一个节点为当前节点
        if len(path) >= 2:
            prior = path[-2]  # 选定当前节点的上一个节点
        else:
            prior = path[-1]
        if len(G.neighbors(cur)) > 0:  # 如果有边与当前节点相连
            if f == 3:
                next_node = prior
                f = 0
            else:
                next_node = np.random.choice(a=list(G.neighbors(cur)), size=1)[0]
                f += 1
            path.append(next_node)  # 随机选取一个作为下一个当前节点
        else:
            break
    return [str(node) for node in path]  # 返回一条随机游走序列

if __name__ == '__main__':
## 算法方案文档待更新
## 这边跟之前人工网络的实验的BMLPA代码已经修改了，主要是init和RC函数
## 密度计算使用邻居度的sigmoid求和
## BMLPA采用异步更新
    p = 0.7
    alpha = 0.8
    x = 5

    # 稠密图alpha往大的找，稀疏图往小了找
    # input_path = "F:\develop\ArtificialNetGenerate\\binary_networks\Release\overlappingNetwork\\network10k-0.3-1000-6.dat"
    # output_path = "F:\develop\ArtificialNetGenerate\\binary_networks\Release\overlappingNetwork\\network.embeddings"
    input_path = "F:\develop\ArtificialNetGenerate\\binary_networks\Release\\ExpTest2\\network3k0.7-90-2.dat"
    output_path = "F:\develop\ArtificialNetGenerate\\binary_networks\Release\\ExpTest2\\Mine-network3k0.7-90-2.embeddings"
    output_filepath = "F:\develop\ArtificialNetGenerate\\binary_networks\Release\\ExpData\\BMLPA_cluster.txt"

    G_nx = load_graph(input_path)

    K = np.max([np.ceil(G_nx.number_of_nodes() * alpha),1])


    # 一般增加p，发现社区数增多，反之减少。
    lable_dict,coms = BMLPA(G_nx, p=p,maxiter=100,flag=True)

    ###################################################################################################
    toComtxt(output_filepath, coms)
    # 计算模块度Q
    partitions = []
    file = open(output_filepath)
    for line in file.readlines():
        partition = line.strip().split(sep=' ')
        partition = list(map(int, partition))  # 将读出来的数据转化成int
        partitions.append(partition)
    print(partitions)
    print('Value of modularity（老师写的,修改后） :', evaluator.cal_EQ(partitions,G_nx).__format__('.4f'))





    print("Random walk based on node density~")
    time1 = time.time()
    pre_walk_length, pre_window_size = random_walk_density_based(G_nx,k=K,N=len(G_nx.nodes()),a=0.5)
    time2 = time.time()
    print("Random walk based on node density Time cost: ", time2 - time1)

    number_walks = 10 # max
    walk_length = pre_walk_length
    representation_size = 128
    window_size = pre_window_size

    workers = 4

    print("Number of nodes: {}".format(len(G_nx.nodes())))

    num_walks = len(G_nx.nodes()) * number_walks
    print("Number of walks: {}".format(num_walks))

    data_size = num_walks * walk_length
    print("Data size (walks*length): {}".format(data_size))

    walks = []
    print("Random Walking...")
    walks_1 = build_my_corpus(G_nx,lable_dict,num_paths=number_walks, path_length=walk_length,x=x)
    walks.extend(walks_1)

    print("Training...")
    time1 = time.time()
    model = Word2Vec(walks, size=representation_size, window=window_size, min_count=0, sg=1, hs=0, workers=workers)
    time2 = time.time()

    print("SkipGram Training Time cost: ",time2-time1)
    model.wv.save_word2vec_format(output_path)
