import json
import requests
import numpy as np
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
        entity_embedding = json.loads(str(response.content, 'utf-8'))['entity_embedding']
        return np.array(entity_embedding)

    def get_entity_embedding_by_eid(self, eid: int) -> np.array:
        params = {'eid': eid}
        response = requests.post("http://" + self.ip_address + ":" + self.port + "/entity_embedding_by_eid/",
                                 json=params,
                                 proxies=self.proxies)
        entity_embedding = json.loads(str(response.content, 'utf-8'))['entity_embedding']
        return np.array(entity_embedding)

    def get_relation_embedding_by_relation(self, relation: str) -> np.array:
        relation = remove_freebase_ns_prefix(relation)
        params = {'relation': relation}
        response = requests.post("http://" + self.ip_address + ":" + self.port + "/relation_embedding_by_relation/",
                                 json=params,
                                 proxies=self.proxies)
        relation_embedding = json.loads(str(response.content, 'utf-8'))['relation_embedding']
        return np.array(relation_embedding)

    def get_relation_embedding_by_rid(self, rid: int) -> np.array:
        params = {'rid': rid}
        response = requests.post("http://" + self.ip_address + ":" + self.port + "/relation_embedding_by_rid/",
                                 json=params,
                                 proxies=self.proxies)
        relation_embedding = json.loads(str(response.content, 'utf-8'))['relation_embedding']
        return np.array(relation_embedding)

    def get_adj_list_by_mid(self, mid: str) -> list:
        mid = remove_freebase_ns_prefix(mid)
        params = {'mid': mid}
        response = requests.post("http://" + self.ip_address + ":" + self.port + "/adj_list/", json=params,
                                 proxies=self.proxies)
        adj_list = json.loads(str(response.content, 'utf-8'))['adj_list']
        return adj_list

    def get_adj_list_by_mid_list(self, mid_list: list) -> list:
        res = []
        for mid in mid_list:
            adj_list = self.get_adj_list_by_mid(mid)
            for triple in adj_list:
                res.append(triple)
        return res

    def get_sop_embedding_list(self, id_triple_list: list):
        subject_list, object_list, predicate_list = get_sop_id_list(id_triple_list)
        subject_embedding = np.zeros((len(subject_list), 50))
        object_embedding = np.zeros((len(object_list), 50))
        predicate_embedding = np.zeros((len(predicate_list), 50))

        for i in range(0, len(subject_list)):
            subject_embedding[i] = self.get_entity_embedding_by_eid(subject_list[i])
            object_embedding[i] = self.get_entity_embedding_by_eid(object_list[i])
            predicate_embedding[i] = self.get_relation_embedding_by_rid(predicate_list[i])
        return subject_embedding, object_embedding, predicate_embedding


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


def get_consecutive_sop_id_list(id_triple_list):
    """
    Get subject, predicate, and object id list from a triple list.
    Note that the id is renumbered from 0 to the number of entity/relation in this list.
    Args:
        id_triple_list: a triple list in the form of entity/relation id.

    Returns: subject list, object list, and predicate list in the form of entity/relation id,
    starting from 0 to the number of entity/relation in this list.
    """
    subject_list = []
    object_list = []
    entity_dict = {}  # original id -> new id starting from 0
    entity_count = 0
    for id_triple in id_triple_list:
        s = id_triple[0]
        o = id_triple[1]
        if not (s in entity_dict):
            entity_dict[s] = entity_count
            entity_count += 1
        if not (o in entity_dict):
            entity_dict[o] = entity_count
            entity_count += 1
        subject_list.append(entity_dict[s])
        object_list.append(entity_dict[o])
    return subject_list, object_list
