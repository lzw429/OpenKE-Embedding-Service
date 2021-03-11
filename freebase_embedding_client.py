import json
import requests
import numpy as np
import torch

from src.retriever.kb_retriever import KBRetriever, remove_freebase_ns_prefix


class FreebaseEmbeddingClient:
    ip_address: str
    port: str

    def __init__(self, ip_address, port):
        self.ip_address = ip_address
        self.port = port
        self.proxies = {"http": None, "https": None}

    def get_entity_embedding_by_mid(self, mid: str) -> np.array:
        mid = remove_freebase_ns_prefix(mid)
        params = {'mid': mid}
        response = requests.post("http://" + self.ip_address + ":" + self.port + "/entity_embedding_by_mid/",
                                 json=params,
                                 proxies=self.proxies)
        if response.status_code == 200:
            entity_embedding = json.loads(str(response.content, 'utf-8'))['entity_embedding']
        else:
            print("[WARN] response status code: " + str(response.status_code) + ", mid: " + mid)
            return zero_embedding()
        return np.array(entity_embedding)

    def get_entity_embedding_by_eid(self, eid: int) -> np.array:
        params = {'eid': eid}
        response = requests.post("http://" + self.ip_address + ":" + self.port + "/entity_embedding_by_eid/",
                                 json=params,
                                 proxies=self.proxies)
        if response.status_code == 200:
            entity_embedding = json.loads(str(response.content, 'utf-8'))['entity_embedding']
        else:
            print("[WARN] response status code: " + str(response.status_code) + ", eid: " + str(eid))
            return zero_embedding()
        return np.array(entity_embedding)

    def get_relation_embedding_by_relation(self, relation: str) -> np.array:
        relation = remove_freebase_ns_prefix(relation)
        params = {'relation': relation}
        response = requests.post("http://" + self.ip_address + ":" + self.port + "/relation_embedding_by_relation/",
                                 json=params,
                                 proxies=self.proxies)
        if response.status_code == 200:
            relation_embedding = json.loads(str(response.content, 'utf-8'))['relation_embedding']
        else:
            print("[WARN] response status code: " + str(response.status_code) + ", relation: " + relation)
            return zero_embedding()
        return np.array(relation_embedding)

    def get_relation_embedding_by_rid(self, rid: int) -> np.array:
        params = {'rid': rid}
        response = requests.post("http://" + self.ip_address + ":" + self.port + "/relation_embedding_by_rid/",
                                 json=params,
                                 proxies=self.proxies)
        if response.status_code == 200:
            relation_embedding = json.loads(str(response.content, 'utf-8'))['relation_embedding']
        else:
            print("[WARN] response status code: " + str(response.status_code) + ", rid: " + str(rid))
            return zero_embedding()
        return np.array(relation_embedding)

    def get_adj_list_by_mid(self, mid: str) -> list:
        mid = remove_freebase_ns_prefix(mid)
        params = {'mid': mid}
        response = requests.post("http://" + self.ip_address + ":" + self.port + "/adj_list/", json=params,
                                 proxies=self.proxies)
        if response.status_code == 200:
            adj_list = json.loads(str(response.content, 'utf-8'))['adj_list']
        else:
            print("[WARN] response status code: " + str(response.status_code) + ", mid: " + str(mid))
            return []
        return adj_list

    def get_inverse_adj_list_by_mid(self, mid: str) -> list:
        mid = remove_freebase_ns_prefix(mid)
        params = {'mid': mid}
        response = requests.post("http://" + self.ip_address + ":" + self.port + "/inverse_adj_list/", json=params,
                                 proxies=self.proxies)
        if response.status_code == 200:
            inverse_adj_list = json.loads(str(response.content, 'utf-8'))['inverse_adj_list']
        else:
            print("[WARN] response status code: " + str(response.status_code) + ", mid: " + str(mid))
            return []
        return inverse_adj_list

    def get_adj_list_by_mid_list(self, mid_list: list) -> list:
        res = []
        for mid in mid_list:
            res += self.get_adj_list_by_mid(mid)
        return res

    def get_inverse_adj_list_by_mid_list(self, mid_list: list) -> list:
        res = []
        for mid in mid_list:
            res += self.get_inverse_adj_list_by_mid(mid)
        return res

    def get_adj_and_inverse_adj_list_by_mid_list(self, mid_list: list) -> list:
        return self.get_adj_list_by_mid_list(mid_list) + self.get_inverse_adj_list_by_mid_list(mid_list)

    def get_entity_id_by_mid(self, mid: str) -> int:
        mid = remove_freebase_ns_prefix(mid)
        params = {'mid': mid}
        response = requests.post("http://" + self.ip_address + ":" + self.port + "/entity_id_by_mid/", json=params,
                                 proxies=self.proxies)
        if response.status_code == 200:
            entity_id = json.loads(str(response.content, 'utf-8'))['entity_id']
        else:
            print("[WARN] response status code: " + str(response.status_code) + ", mid: " + str(mid))
            return -1
        return entity_id

    def get_entity_id_list_by_mid_list(self, mid_list: list) -> list:
        res = []
        for mid in mid_list:
            res.append(self.get_entity_id_by_mid(mid))
        return res

    def get_relation_id_by_relation(self, relation: str) -> list:
        relation = remove_freebase_ns_prefix(relation)
        params = {'relation': relation}
        response = requests.post("http://" + self.ip_address + ":" + self.port + "/relation_id_by_relation/",
                                 json=params,
                                 proxies=self.proxies)
        if response.status_code == 200:
            relation_id = json.loads(str(response.content, 'utf-8'))['relation_id']
        else:
            print("[WARN] response status code: " + str(response.status_code) + ", relation: " + str(relation))
            return -1
        return relation_id

    def get_dgl_graph(self, id_triple_list: list, ans_mid_list: list) -> (
            list, list, np.array, np.array, np.array):
        """
        Get subject, predicate, and object id list from a triple list.
        Note that the id is renumbered from 0 to the number of entity/relation in this list.
        Args:
            id_triple_list: a triple list in the form of entity/relation id.
            ans_mid_list: a list of answers for a question, used for node label

        Returns: subject list, object list in the form of new entity/relation id,
        starting from 0 to the number of entity/relation in this list, and corresponding entity/relation embedding and labels
        """
        entity_list = []
        relation_list = []  # original id
        subject_new_list = []
        object_new_list = []  # new id
        entity_dict = {}
        relation_dict = {}  # original id -> new id starting from 0
        entity_count = 0
        relation_count = 0
        label_list = []

        for id_triple in id_triple_list:
            (s, o, p) = id_triple
            if not (s in entity_dict):
                entity_dict[s] = entity_count
                entity_count += 1
                entity_list.append(s)
            if not (o in entity_dict):
                entity_dict[o] = entity_count
                entity_count += 1
                entity_list.append(o)
            if not (p in relation_dict):
                relation_dict[p] = relation_count
                relation_count += 1
                relation_list.append(p)
            subject_new_list.append(entity_dict[s])
            object_new_list.append(entity_dict[o])

        entity_embedding = np.zeros((entity_count, 50), dtype=np.float32)
        relation_embedding = np.zeros((len(id_triple_list), 50))
        for i in range(len(entity_list)):
            entity_embedding[i] = self.get_entity_embedding_by_eid(entity_list[i])
        for i in range(len(id_triple_list)):
            relation_embedding[i] = self.get_relation_embedding_by_rid(id_triple_list[i][2])

        ans_entity_id_list = self.get_entity_id_list_by_mid_list(ans_mid_list)
        for entity_id in entity_list:
            if entity_id in ans_entity_id_list:
                label_list.append(1)
            else:
                label_list.append(0)
        return subject_new_list, object_new_list, entity_embedding, relation_embedding, np.array(label_list)


def zero_embedding():
    return np.zeros(50)


def get_sop_id_list(id_triple_list: list) -> (list, list, list):
    """
    Get subject, predicate, and object id list from a triple list.
    Note that this id is the OpenKE entity/relation serial number for Freebase.
    Args:
        id_triple_list: a triple list in the form of entity/relation id.

    Returns: subject list, object list, and predicate list in the form of entity/relation id.

    """
    subject_list = []
    object_list = []
    predicate_list = []
    for id_triple in id_triple_list:
        subject_list.append(id_triple[0])
        object_list.append(id_triple[1])
        predicate_list.append(id_triple[2])
    return subject_list, object_list, predicate_list
