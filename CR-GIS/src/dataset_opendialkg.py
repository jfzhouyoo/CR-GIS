import enum
import numpy as np
from torch import int64, inverse, long
from tqdm import tqdm
from copy import deepcopy
import pickle
import json
from nltk import word_tokenize
from torch.utils.data.dataset import Dataset
import numpy as np
from copy import deepcopy
import random

class dataset_opendialkg(object):
    def __init__(self,filename, opt):
        self.entity2id = pickle.load(open("../data/opendialkg/entity2id.pkl","rb"))
        self.entity_max = len(self.entity2id) # 100925
        
        self.id2entity = pickle.load(open("../data/opendialkg/id2entity.pkl","rb"))
        self.relation2id = pickle.load(open("../data/opendialkg/relation2id.pkl","rb"))
        # self.id2relation = pickle.load(open("../data/opendialkg/id2relation.pkl","rb"))
        self.opendialkg = pickle.load(open("../data/opendialkg/kg.pkl","rb"))
        self.text_dict = pickle.load(open("../data/opendialkg/text_dict.pkl","rb"))

        self.batch_size = opt["batch_size"]
        self.max_context_length = opt["max_context_length"]
        self.histoty_window_size = opt["history_window_size"]
        self.max_response_length = opt["max_response_length"]
        self.max_uttr_num = opt["history_window_size"]
        self.max_uttr_length = opt["max_uttr_length"]
        self.max_uttr_entity_num = opt["max_uttr_entity_num"]
        self.max_entity_num = opt["max_entity_num"]
        self.entity_num = opt["entity_num"]
        self.filename = f"../data/opendialkg/{filename}"
        self.word2index = json.load(open("../data/opendialkg/word2index.json",encoding="utf-8"))
        self.word2index["_split_"] = len(self.word2index)+5
        self.unk = 3

        self.prepare_word2vec()
    
    def prepare_word2vec(self):
        """
        get word token embedding from RoBerta, 添加 _split_ token
        """
        # Done in create_word2vec.py 
        pass

    def data_process(self):
        """
        """
        with open(self.filename,"r") as f:
            content = json.load(f)
        print("length of "+ self.filename,": ", len(content))

        data_set_original = []
        no_response_count = 0
        for line in tqdm(content, total=len(content)):
            dialogue = line["dialogue"]
            dial_id = line["dial_id"]
            previous_sentence = ""
            dialogue_history_utterance = []
            dialogue_history = []
            entities = []
            context_uttr_history = []
            context_uttr_entity_history = []
            response = ""
            for ti, turn in enumerate(dialogue):
                # 先看是包含路径的metadata
                if 'action_id' in turn and turn['action_id'] == 'kgwalk/choose_path':
                    kg_path = turn['metadata']['path'][1]
                    kg_path_id = [(self.entity2id[triple[0]], self.relation2id[triple[1]], self.entity2id[triple[2]]) for triple in kg_path]
                    path_utterance = turn['metadata']['path'][2]

                    # 查看metadata之后是否有message，若没有则无response，忽略该条数据
                    ri = ti+1
                    while ri < len(dialogue):
                        if "message" not in dialogue[ri]:
                            ri += 1
                        else:
                            break
                    if ri >= len(dialogue):
                        no_response_count += 1
                        # print(dial_id)
                        continue
                    response_utterance = dialogue[ri]["message"]
                    response = [self.word2index[word] for word in word_tokenize(response_utterance)]

                    target_rec = [self.entity2id[e] for e in self.text_dict[response_utterance]]
                    if kg_path_id[-1][-1] not in target_rec:
                        target_rec.append(kg_path_id[-1][-1])
                        
                    if len(entities)>0:
                        entities_id = [self.entity2id[e] for e in entities]
                    else:
                        entities_id = []

                    # 保存数据
                    sample = {
                        "dial_id": dial_id,
                        "sample_id": len(data_set_original),
                        "response_utterance": response_utterance,
                        "response": response,
                        "dialogue_history_utterance": dialogue_history_utterance[-self.histoty_window_size:],
                        "dialogue_history": dialogue_history[-self.histoty_window_size:],
                        # "history_entities": deepcopy(entities),
                        "history_entities_id": deepcopy(entities_id),
                        "movie_target": [kg_path_id[-1][-1]],
                        "total_target": deepcopy(target_rec),
                        "context_uttr_history": deepcopy(context_uttr_history[-self.max_uttr_num:]),
                        "context_uttr_entity_history": deepcopy(context_uttr_entity_history[-self.max_uttr_num:]),
                    }
                    # 过滤第一条就选择路径的数据
                    if ti != 0:
                        data_set_original.append(sample)
                    
                # 不是包含路径的metadata 忽略
                elif 'action_id' in turn and turn['action_id'] == 'meta_thread/send_meta_message':  # useless
                    previous_sentence = ''
                    pass

                # 正常的message
                else:
                    previous_sentence = turn["message"]
                    if len(previous_sentence) != 0:
                        entities.extend(self.text_dict[previous_sentence])
                        context_uttr_entity_history.append([self.entity2id[e] for e in self.text_dict[previous_sentence]]) # 11.22
                        # entities = list(set(entities)) # 11.22
                        dialogue_history_utterance.append(previous_sentence)
                        dialogue_history.append([self.word2index.get(word, self.unk) for word in word_tokenize(previous_sentence)])
                        context_uttr_history.append([self.word2index.get(word, self.unk) for word in word_tokenize(previous_sentence)])

        data_set = []
        for line in data_set_original:
            context, context_length = self.padding_context(line["dialogue_history_utterance"])
            response, response_length = self.padding_w2v(word_tokenize(line["response_utterance"]), self.max_response_length)
            mask_response, mask_response_length = self.unk_mask(response, response_length)
            assert len(context) == self.max_context_length
            
            context_uttr_list, context_uttr_num = self.padding_hierachical_context(line["context_uttr_history"])
            context_uttr_entity_list, context_uttr_entity_num = self.padding_hierachical_context_uttr_entity(line["context_uttr_entity_history"])
            
            history_entities = line["history_entities_id"]
            total_target = line["total_target"]
            total_target = line["movie_target"]
            negative_target = []
            for idx in range(len(total_target)):
                negative_sample = np.random.randint(1, self.entity_num+1)
                while negative_sample in total_target or negative_sample in negative_target:
                    negative_sample = np.random.randint(1, self.entity_num+1)
                negative_target.append(negative_sample)
            assert len(negative_target) == len(total_target)
            
            rec = 0
            if len(total_target)>0:
                rec = 1
            
            data_set.append([line["dial_id"],
                             line["sample_id"],
                             line["sample_id"],
                             len(data_set),
                             np.array(context), context_length,
                             np.array(response), response_length,
                             np.array(mask_response), mask_response_length,
                             history_entities, history_entities, history_entities,
                             total_target, total_target, total_target, negative_target,
                             rec, rec, rec, rec, rec, rec,
                             np.array(context_uttr_list), context_uttr_num,
                             np.array(context_uttr_entity_list), np.array(context_uttr_entity_num)])
        return data_set
    
    def padding_context(self, context, pad=0, transformer=True):
        vectors = []
        vec_length = []
        if transformer==False:
            pass
        else:
            contexts_combine = []
            for sentence in context[-self.histoty_window_size:-1]:
                contexts_combine.extend(word_tokenize(sentence))
                contexts_combine.append("_split_")
            contexts_combine.extend(word_tokenize(context[-1]))
            context_vector, context_vector_len = self.padding_w2v(contexts_combine, self.max_context_length, transformer)
            return context_vector, context_vector_len

    def padding_w2v(self, sentence, max_length, transformer=True, pad=0, end=2, unk=3):
        vector = [self.word2index.get(word, unk) for word in sentence]
        vector.append(end)

        if len(vector)>max_length:
            if transformer:
                return vector[-max_length:],max_length
            else:
                return vector[:max_length],max_length
        else:
            length = len(vector)
            return vector+(max_length-len(vector))*[pad], length

    def padding_hierachical_context(self, context, pad=0, bos=1, eos=2, unk=3, cls=4):
        original_context = deepcopy(context)
        context_uttr_list = []
        context_uttr_num = len(context)
        for uttr in original_context:
            if len(uttr)>self.max_uttr_length-1:
                uttr = uttr[-(self.max_uttr_length-1):]
                uttr = [cls] + uttr # 起始位置加 bos 类似 CLS 效果
            else:
                uttr = [cls] + uttr # 起始位置加 bos 类似 CLS 效果
                uttr = uttr + [pad] * (self.max_uttr_length-len(uttr))
            context_uttr_list.append(uttr)
            assert len(uttr) == self.max_uttr_length # 保证所有 uttr 长度一致
        while len(context_uttr_list)<self.max_uttr_num:
            context_uttr_list.append([pad]*self.max_uttr_length)
        return context_uttr_list, context_uttr_num
    
    def padding_hierachical_context_uttr_entity(self, context_uttr_entity, padding_idx=0):
        # context_uttr_entity_num = len(context_uttr_entity)
        original_context_uttr_entity = deepcopy(context_uttr_entity)
        context_uttr_entity_list = []
        context_uttr_entity_num = []
        for uttr_entity in original_context_uttr_entity:
            context_uttr_entity_num.append(len(uttr_entity))
            if len(uttr_entity)>self.max_uttr_entity_num:
                uttr_entity = uttr_entity[-self.max_uttr_entity_num:]
            else:
                uttr_entity = uttr_entity + [padding_idx] * (self.max_uttr_entity_num-len(uttr_entity))
            context_uttr_entity_list.append(uttr_entity)
            assert len(uttr_entity) == self.max_uttr_entity_num
        while len(context_uttr_entity_list)<self.max_uttr_num:
            context_uttr_entity_list.append([padding_idx]*self.max_uttr_entity_num)
            context_uttr_entity_num.append(padding_idx)
        assert len(context_uttr_entity_num) == len(context_uttr_entity_list)
        return context_uttr_entity_list, context_uttr_entity_num
    
    def unk_mask(self, sentence, length, pad=0, end=2, unk=3):
        unk_sentence = deepcopy(sentence)
        length = deepcopy(length)
        for i, idx in enumerate(sentence):
            if idx != end:
                unk_sentence[i] = unk
            else:
                break
        return unk_sentence, length
    
    def padding_path(self, kg_path_id, hop=2):
        kg_path_padding = []
        assert len(kg_path_id) <= hop
        while len(kg_path_id)<hop:
            kg_path_id.append((kg_path_id[-1][2], self.relation2id["self_loop"], kg_path_id[-1][2]))
        assert len(kg_path_id) == hop
        kg_path_padding.append((self.relation2id["self_loop"], kg_path_id[0][0]))
        for kg_path in kg_path_id:
            kg_path_padding.append((kg_path[1], kg_path[2]))
        return kg_path_padding
    
    def trans_path_utterance(self, path_utterance, end=2, unk=3):
        vector = [self.word2index.get(word, unk) for word in path_utterance]
        vector.append(end)
        return vector
        
        
class OpenDialKG(object):
    def __init__(self, dataset):
        self.data = dataset
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index, align_context=50, align_response=10):
        '''
            line["user"], 
            line["conv_id"],
            line["local_id"], 
            len(data_set), 
            np.array(context), context_length,
            np.array(response), response_length,
            np.array(mask_response), mask_response_length,
            target_history, entities_history, total_entities_history,
            movie_target, topic_target, total_target, negative_target,
            recommender_rec_movie, recommender_rec_topic, recommender_rec,
            rec_movie,rec_topic, rec, context_uttr_list, context_uttr_num,
            context_uttr_entity_list, context_uttr_entity_num
        '''
        user, conv_id, local_id, sample_id, context, context_length, response, response_length, mask_response, mask_response_length, target_history, entities_history, total_entities_history, movie_target, topic_target, total_target, negative_target, recommender_rec_movie, recommender_rec_topic, recommender_rec, rec_movie, rec_topic, rec, context_uttr_list, context_uttr_num, context_uttr_entity_list, context_uttr_entity_num = self.data[index]
        
        target_history = np.array(target_history+[0]*(align_context-len(target_history)))
        entities_history = np.array(entities_history+[0]*(align_context-len(entities_history)))
        total_entities_history = np.array(total_entities_history+[0]*(align_context-len(total_entities_history)))
        total_entities_history_mask = np.array([1]*len(total_entities_history)+[0]*(align_context-len(total_entities_history)))

        movie_target = np.array(movie_target+[0]*(align_response-len(movie_target)))
        topic_target = np.array(topic_target+[0]*(align_response-len(topic_target)))
        total_target = np.array(total_target+[0]*(align_response-len(total_target)))
        negative_target = np.array(negative_target+[0]*(align_response-len(negative_target)))
        
        return [user, conv_id, local_id, sample_id, context, context_length, response, response_length, mask_response, mask_response_length, target_history, entities_history, total_entities_history, total_entities_history_mask, movie_target, topic_target, total_target, negative_target, recommender_rec_movie, recommender_rec_topic, recommender_rec, rec_movie, rec_topic, rec, context_uttr_list, context_uttr_num, context_uttr_entity_list, context_uttr_entity_num]

    def dataset2list(self, align_context=50, align_response=10):
        dataset = []
        for index in range(len(self.data)):
            user, conv_id, local_id, sample_id, context, context_length, response, response_length, mask_response, mask_response_length, target_history, entities_history, total_entities_history, movie_target, topic_target, total_target, negative_target, recommender_rec_movie, recommender_rec_topic, recommender_rec, rec_movie, rec_topic, rec, context_uttr_list, context_uttr_num, context_uttr_entity_list, context_uttr_entity_num = self.data[index]
        
            target_history = np.array(target_history+[0]*(align_context-len(target_history)))
            entities_history = np.array(entities_history+[0]*(align_context-len(entities_history)))
            total_entities_history = np.array(total_entities_history+[0]*(align_context-len(total_entities_history)))
            total_entities_history_mask = np.array([1]*len(total_entities_history)+[0]*(align_context-len(total_entities_history)))

            movie_target = np.array(movie_target+[0]*(align_response-len(movie_target)))
            topic_target = np.array(topic_target+[0]*(align_response-len(topic_target)))
            total_target = np.array(total_target+[0]*(align_response-len(total_target)))
            negative_target = np.array(negative_target+[0]*(align_response-len(negative_target)))

            dataset.append([user, conv_id, local_id, sample_id, context, context_length, response, response_length, mask_response, mask_response_length, target_history, entities_history, total_entities_history, total_entities_history_mask, movie_target, topic_target, total_target, negative_target, recommender_rec_movie, recommender_rec_topic, recommender_rec, rec_movie, rec_topic, rec, context_uttr_list, context_uttr_num, context_uttr_entity_list, context_uttr_entity_num])
        return dataset


class Opendial(Dataset):
    def __init__(self, dataset, entity_num):
        self.data=dataset
        self.entity_num = entity_num

    def __getitem__(self, index):
        '''
        ([line["dial_id"],line["sample_id"],np.array(context),context_length,np.array(response),response_length,np.array(mask_response),mask_response_length,
                             kg_path_padding,kg_path_padding[-1][-1],line["starting_entity_id"],line["history_entities"]])
        '''
        dial_id, sample_id, context, context_length, response, response_length,mask_response, mask_response_length, kg_path_padding, rec, start_entity, entities, path_uttr, path_fake_uttr = self.data[index]
        # entity_vec = np.zeros(self.entity_num)
        entities_vector=np.zeros(50,dtype=np.int)
        point=0
        for en in entities:
            # entity_vec[en]=1
            entities_vector[point]=en
            point+=1
        path_uttr_tmp = deepcopy(path_uttr)
        path_fake_uttr_tmp = deepcopy(path_fake_uttr)
        assert len(path_uttr_tmp)<200
        while len(path_uttr_tmp)<200:
            path_uttr_tmp.append(0)
        path_uttr_vector = np.array(path_uttr_tmp)
        assert len(path_fake_uttr_tmp)<200
        while len(path_fake_uttr_tmp)<200:
            path_fake_uttr_tmp.append(0)
        path_fake_uttr_vector=np.array(path_fake_uttr_tmp)

        return dial_id, sample_id, context, context_length, response, response_length,mask_response, mask_response_length, rec, start_entity, entities_vector, path_uttr_vector, path_fake_uttr_vector

    def __len__(self):
        return len(self.data)      



if __name__ == '__main__':
    pass
