import json

"""
从训练好的模型保存下的权重里，提取出entity和relation的embedding
"""
dataset = "FB15K237"  # humans_wikidata
model_name = "transe"  # transE  rotate  distmult  complEx


f1 = open("./checkpoint/{}/{}/entity_vec.{}".format(dataset, model_name, model_name), "w")
f2 = open("./checkpoint/{}/{}/relation_vec.{}".format(dataset, model_name, model_name), "w")

with open("./checkpoint/{}/{}/{}_emb.vec".format(dataset, model_name, model_name)) as f:
    res =json.load(f)
    str1 = ""
    str2 = ""
    list1 = res["ent_embeddings.weight"]
    list2 = res["rel_embeddings.weight"]

    entity_num = len(list1)
    entity_emb_dim = len(list1[0])
    relation_num = len(list2)
    relation_emb_dim = len(list2[0])
    print(f'entity_num:{entity_num}, entity_emb_num:{entity_emb_dim}')
    print(f'relation_num:{relation_num}, relation_emb_dim:{relation_emb_dim}')
    f1.write(str(entity_num) + ' ' + str(entity_emb_dim) + '\n')
    f2.write(str(relation_num) + ' ' + str(relation_emb_dim) + '\n')

    for i, temp_enemb in enumerate(list1):
        str1 += str(i) + " "
        # str1 += str(list(list1[i])) + "\n"
        list1_str = ' '.join(map(str, list1[i]))
        str1 += list1_str + "\n"
    for i, temp_relenm in enumerate(list2):
        str2 += str(i) + " "
        # str2 += str(list(list2[i])) + "\n"
        list2_str = ' '.join(map(str, list2[i]))
        str2 += list2_str + "\n"

    f1.write(str1)
    f2.write(str2)
    f1.close()
    f2.close()