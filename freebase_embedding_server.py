import numpy as np
import datetime
from flask import Flask, request, jsonify, json


class FreebaseEmbeddingServer:
    dir_path: str
    entity_to_id: dict  # entity mid -> entity id
    relation_to_id: dict
    entity_vec: np.memmap
    relation_vec: np.memmap
    dim: int  # embedding dimension for each entity or relation
    id_adj_list: dict  # adjacency list
    id_inverse_adj_list: dict  # inverse adjacency list

    def __init__(self, freebase_embedding_dir_path):
        start_time = datetime.datetime.now()
        print("[INFO] Loading OpenKE TransE for Freebase...")

        # file paths
        self.dir_path = freebase_embedding_dir_path.rstrip("/").rstrip("\\")
        entity_emb_filepath = self.dir_path + "/embeddings/dimension_50/transe/entity2vec.bin"
        relation_emb_filepath = self.dir_path + "/embeddings/dimension_50/transe/relation2vec.bin"
        entity_to_id_filepath = self.dir_path + "/knowledge_graphs/entity2id.txt"
        relation_to_id_filepath = self.dir_path + "/knowledge_graphs/relation2id.txt"
        triple_to_id_filepath = self.dir_path + "/knowledge_graphs/triple2id.txt"

        # initialize variables
        self.entity_to_id = dict()
        self.relation_to_id = dict()
        self.id_adj_list = dict()
        self.id_inverse_adj_list = dict()
        self.entity_vec = np.memmap(entity_emb_filepath, dtype='float32', mode='r')
        self.relation_vec = np.memmap(relation_emb_filepath, dtype='float32', mode='r')
        self.dim = 50

        # build self.entity_to_id
        entity_to_id_file = open(entity_to_id_filepath)
        for line in entity_to_id_file.readlines():
            line.rstrip("\n")
            if "\t" in line:
                line_split = line.split("\t")
            elif " " in line:
                line_split = line.split(" ")
            else:
                continue
            self.entity_to_id[line_split[0]] = line_split[1]
        entity_to_id_file.close()

        # build self.relation_to_id
        relation_to_id_file = open(relation_to_id_filepath)
        for line in relation_to_id_file.readlines():
            line.rstrip("\n")
            if "\t" in line:
                line_split = line.split("\t")
            elif " " in line:
                line_split = line.split(" ")
            else:
                continue
            self.relation_to_id[line_split[0]] = line_split[1]
        relation_to_id_file.close()

        # build adj_list and inverse_adj_list
        triple_to_id_file = open(triple_to_id_filepath)
        for line in triple_to_id_file.readlines():
            line.rstrip("\n")
            if "\t" in line:
                line_split = line.split("\t")
            elif " " in line:
                line_split = line.split(" ")
            else:
                continue
            subject_id = int(line_split[0])
            object_id = int(line_split[1])
            predicate_id = int(line_split[2])

            # for adj list
            if not (subject_id in self.id_adj_list.keys()):
                self.id_adj_list[subject_id] = []
            self.id_adj_list[subject_id].append((subject_id, object_id, predicate_id))
            # for inverse adj list
            if not (object_id in self.id_inverse_adj_list.keys()):
                self.id_inverse_adj_list[object_id] = []
            self.id_inverse_adj_list[object_id].append((subject_id, object_id, predicate_id))

        triple_to_id_file.close()

        print("[INFO] OpenKE TransE for Freebase has been loaded")
        print("[INFO] time consumed: " + str(datetime.datetime.now() - start_time))

    def get_entity_id_by_mid(self, mid: str) -> int:
        return self.entity_to_id[mid]

    def get_relation_id_by_relation(self, relation: str) -> int:
        return self.relation_to_id[relation]

    def get_entity_embedding_by_mid(self, mid: str):
        return self.get_entity_embedding_by_eid(int(self.entity_to_id[mid]))

    def get_entity_embedding_by_eid(self, idx: int):
        return self.entity_vec[self.dim * idx:self.dim * (idx + 1)]

    def get_relation_embedding_by_relation(self, relation: str):
        return self.get_relation_embedding_by_rid(int(self.relation_to_id[relation]))

    def get_relation_embedding_by_rid(self, idx: int):
        return self.relation_vec[self.dim * idx:self.dim * (idx + 1)]

    def get_adj_list(self, mid: str):
        idx = int(self.entity_to_id[mid])
        if idx in self.id_adj_list:
            return self.id_adj_list[idx]
        return None

    def get_inverse_adj_list(self, mid: str):
        idx = int(self.entity_to_id[mid])
        if idx in self.id_inverse_adj_list:
            return self.id_inverse_adj_list[idx]
        return None


app = Flask(__name__)
service = FreebaseEmbeddingServer("/home2/yhshu/yhshu/workspace/Freebase")


@app.route('/entity_embedding_by_mid/', methods=['POST'])
def entity_embedding_by_mid_service():
    params = json.loads(request.data.decode("utf-8"))
    res = service.get_entity_embedding_by_mid(params['mid']).tolist()
    return jsonify({'entity_embedding': res})


@app.route('/entity_embedding_by_eid/', methods=['POST'])
def entity_embedding_by_eid_service():
    params = json.loads(request.data.decode("utf-8"))
    res = service.get_entity_embedding_by_eid(params['eid']).tolist()
    return jsonify({'entity_embedding': res})


@app.route('/relation_embedding_by_relation/', methods=['POST'])
def relation_embedding_by_relation_service():
    params = json.loads(request.data.decode("utf-8"))
    res = service.get_relation_embedding_by_relation(params['relation']).tolist()
    return jsonify({'relation_embedding': res})


@app.route('/relation_embedding_by_rid/', methods=['POST'])
def relation_embedding_by_rid_service():
    params = json.loads(request.data.decode("utf-8"))
    res = service.get_relation_embedding_by_rid(params['rid']).tolist()
    return jsonify({'relation_embedding': res})


@app.route('/adj_list/', methods=['POST'])
def adj_list_service():
    params = json.loads(request.data.decode("utf-8"))
    res = service.get_adj_list(params['mid'])
    return jsonify({'adj_list': res})


@app.route('/inverse_adj_list/', methods=['POST'])
def inverse_adj_list_service():
    params = json.loads(request.data.decode("utf-8"))
    res = service.get_inverse_adj_list(params['mid'])
    return jsonify({'inverse_adj_list': res})


@app.route('/entity_id_by_mid/', methods=['POST'])
def entity_id_by_mid_service():
    params = json.loads(request.data.decode("utf-8"))
    res = service.get_entity_id_by_mid(params['mid'])
    return jsonify({'entity_id': res})


@app.route('/relation_id_by_relation/', methods=['POST'])
def relation_id_by_relation_service():
    params = json.loads(request.data.decode("utf-8"))
    res = service.get_relation_id_by_relation(params['relation'])
    return jsonify({'relation_id': res})


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=8898)  # '0.0.0.0' is necessary for visibility
