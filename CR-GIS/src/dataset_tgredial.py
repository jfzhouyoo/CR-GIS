import pickle
from numpy.lib.arraysetops import isin
from tqdm import tqdm
from copy import deepcopy
import json
from torch.utils.data.dataset import Dataset
import numpy as np
import random
import gensim
import argparse
import json

class dataset_tgredial(object):
    def __init__(self, corpus_filename, opt):
        self.entity2id = pickle.load(open(f"../data/tgredial/entity2id.pkl", "rb"))
        self.topic2id = pickle.load(open("../data/tgredial/topic2id.pkl", "rb"))
        self.kg = pickle.load(open("../data/tgredial/kg.pkl", "rb"))
        self.token2id = pickle.load(open("../data/tgredial/word2id.pkl", "rb"))
        self.token2id["__split__"] = len(self.token2id)

        self.corpus_filename = f"../data/tgredial/{corpus_filename}"
        self.opt = opt

        self.batch_size = opt["batch_size"]
        self.max_context_length = opt["max_context_length"]
        self.max_response_length = opt["max_response_length"]
        self.history_window_size = opt["history_window_size"]
        self.max_uttr_num = opt["history_window_size"]
        self.max_uttr_length = opt["max_uttr_length"]
        self.max_uttr_entity_num = opt["max_uttr_entity_num"]
        self.max_entity_num = opt["max_entity_num"]
        self.entity_num = opt["entity_num"]
        
        self.prepare_word2vec()
    
    def prepare_word2vec(self):
        '''
        get word token embedding from gensim, add _split_ token
        Done in build_word2vec.py
        '''
        pass

    def data_process(self):
        '''
        process data
        '''
        with open(self.corpus_filename,encoding="utf-8") as f:
            corpus = json.load(f)
        print("*"*10+" processing "+self.corpus_filename+" ......")
        
        data_set_original = []
        for line in tqdm(corpus, total=len(corpus)):
            user_id = line["user_id"]
            conv_id = line["conv_id"]
            session_num = len(line["messages"])
            context_list = []
            dialogue_history = []
            entities_history = []
            target_history = []
            total_entities_history = []
            response = ""
            context_uttr_history = []
            context_uttr_entity_history = []
            for session_idx, message in enumerate(line["messages"]):
                local_id = message["local_id"]
                role = message["role"]
                target = message["target"]
                movie = message["movie"]
                text = message["text"]
                entity = message["entity"]
                word = message["word"]

                if len(context_list)>0:
                    '''保证上一句与当前句是两个角色，否则再进行更细化处理，已验证上下句均为两个角色，这里再次确认'''
                    assert role != context_list[-1]["role"]
                context_list.append(message) # save current session info

                # process target
                session_target = []
                reject_target = [] # save feedback
                assert len(target) <= 5
                if len(target)>0:
                    session_target, reject_target = self.process_target(target)
                # for t in session_target:
                #     if t not in target_history: # De-duplication
                #         target_history.append(t)
                target_history.extend(session_target) # Don't De-duplication
                # remove element in reject_target
                if len(reject_target)>0:
                    target_temp = [t for t in target_history if t not in reject_target]
                    target_history = deepcopy(target_temp)

                # process movie, entity
                session_entities = []
                movie_entity = movie + entity
                if len(movie_entity)>0:
                    for e in movie_entity:
                        if e in self.entity2id:
                            session_entities.append(self.entity2id[e])
                session_entities = list(set(session_entities))
                entities_history.extend(session_entities) # Don't De-duplication

                # merge target and movie_entity, De-duplication
                total_entities = []
                for entity in session_target+session_entities:
                    if entity not in total_entities:
                        total_entities.append(entity)
                total_entities_history.extend(total_entities) # Don't De-duplication
                context_uttr_entity_history.append(total_entities) # 11.14

                # process text
                session_text = []
                for token in text:
                    if token in self.token2id:
                        session_text.append(self.token2id[token])
                    else:
                        session_text.append(self.token2id["unk"])
                dialogue_history.append(session_text) # add context history
                context_uttr_history.append(session_text) # 11.14
                
                # obtain response and next targets
                # assert local_id == session_idx+1
                if local_id < session_num and session_idx+1 < session_num: 
                    next_role = line["messages"][session_idx+1]["role"]
                    # next response
                    response_text = line["messages"][session_idx+1]["text"]
                    # response = [self.token2id.get(token, self.token2id["unk"]) for token in response_text] # high coding
                    response = []
                    for token in response_text: # low coding
                        if token in self.token2id:
                            response.append(self.token2id[token])
                        else:
                            response.append(self.token2id["unk"])
                    # next targets
                    movie_target = list(set([self.entity2id[movie] for movie in line["messages"][session_idx+1]["movie"] if movie in self.entity2id]))
                    topic_target, topic_reject_target = self.process_target(line["messages"][session_idx+1]["target"]) 
                    total_target = []
                    for next_target in movie_target+topic_target:
                        if next_target not in total_target:
                            total_target.append(next_target)
                
                    if len(total_entities_history)==0 or len(total_target)==0: 
                        continue 
                    
                    # save data
                    sample = {
                        "user": int(user_id),
                        "conv_id": int(conv_id),
                        "local_id": local_id,
                        "role": role,
                        "dialogue_history": deepcopy(dialogue_history),
                        "target_history": deepcopy(target_history),
                        "entities_history": deepcopy(entities_history),
                        "total_entities_history": deepcopy(total_entities_history),
                        "next_role": next_role,
                        "response": response,
                        "movie_target": movie_target,
                        "topic_target": topic_target,
                        "total_target": total_target,
                        "context_uttr_history": deepcopy(context_uttr_history[-self.max_uttr_num:]),
                        "context_uttr_entity_history": deepcopy(context_uttr_entity_history[-self.max_uttr_num:])
                    }
                    data_set_original.append(sample)

        print("*"*10+" preprocessing done! "+"*"*10)
        
        data_set = []
        for line in tqdm(data_set_original):
            context, context_length = self.padding_context(line["dialogue_history"])
            response, response_length = self.padding_vec(line["response"], self.max_response_length)
            mask_response, mask_response_length = self.unk_mask(response, response_length)
            assert len(context) == self.max_context_length
            assert len(response) == self.max_response_length
            
            context_uttr_list, context_uttr_num = self.padding_hierachical_context(line["context_uttr_history"])
            context_uttr_entity_list, context_uttr_entity_num = self.padding_hierachical_context_uttr_entity(line["context_uttr_entity_history"])
            # assert context_uttr_num == context_uttr_entity_num
            assert len(context_uttr_list) == len(context_uttr_entity_list)
            
            next_role = line["next_role"]
            movie_target = line["movie_target"]
            topic_target = line["topic_target"]
            total_target = line["total_target"]

            if self.opt["train_rec_only"]:
                total_target = line["movie_target"]

            # negative_target = list(np.random.randint(1, self.entity_num+1))
            # assert negative_target[0] not in total_target
            negative_target = []
            for idx in range(len(total_target)):
                negative_sample = np.random.randint(1, self.entity_num+1)
                while negative_sample in total_target or negative_sample in negative_target:
                    negative_sample = np.random.randint(1, self.entity_num+1)
                negative_target.append(negative_sample)
            assert len(negative_target) == len(total_target)
            recommender_rec_movie = 0
            recommender_rec_topic = 0
            recommender_rec = 0
            if next_role == "Recommender":
                if len(movie_target)>0:
                    recommender_rec_movie = 1
                if len(topic_target)>0:
                    recommender_rec_topic = 1
                if len(total_target)>0:
                    recommender_rec = 1
            rec_movie = 0
            rec_topic = 0
            rec = 0
            if len(movie_target)>0:
                rec_movie = 1
            if len(topic_target)>0:
                rec_topic = 1
            if len(total_target)>0:
                rec = 1
            target_history = line["target_history"]
            entities_history = line["entities_history"]
            total_entities_history = line["total_entities_history"]
            if self.opt["train_rec_only"]:
                if len(movie_target) > 0:
                    data_set.append([line["user"], 
                             line["conv_id"],
                             line["local_id"], 
                             len(data_set), 
                             np.array(context), context_length,
                             np.array(response), response_length,
                             np.array(mask_response), mask_response_length,
                             target_history, entities_history, total_entities_history,
                             movie_target, topic_target, total_target, negative_target,
                             recommender_rec_movie, recommender_rec_topic, recommender_rec,
                             rec_movie,rec_topic, rec,
                             np.array(context_uttr_list), context_uttr_num,
                             np.array(context_uttr_entity_list), np.array(context_uttr_entity_num)])
            else:
                data_set.append([line["user"], 
                                line["conv_id"],
                                line["local_id"], 
                                len(data_set), 
                                np.array(context), context_length,
                                np.array(response), response_length,
                                np.array(mask_response), mask_response_length,
                                target_history, entities_history, total_entities_history,
                                movie_target, topic_target, total_target, negative_target,
                                recommender_rec_movie, recommender_rec_topic, recommender_rec,
                                rec_movie,rec_topic, rec,
                                np.array(context_uttr_list), context_uttr_num,
                                np.array(context_uttr_entity_list), np.array(context_uttr_entity_num)])
        print("*"*10+" Re-preprocessing done! "+"*"*10)
        
        return data_set
    
    def padding_context(self, context, transformer=True, pad=0, bos=1, eos=2, unk=3):
        vectors = []
        vec_length = []
        if transformer==False:
            pass
        else:
            context_combine = []
            for sentence in context[-self.history_window_size:-1]:
                context_combine.extend(sentence)
                context_combine.append(self.token2id["__split__"])
            context_combine.extend(context[-1])
            context_vector, context_vector_len = self.padding_vec(context_combine, self.max_context_length)
            return context_vector, context_vector_len

    def padding_vec(self, sentence, max_length, transformer=True, pad=0, bos=1, eos=2, unk=3):
        vector = deepcopy(sentence)
        vector.append(eos)
        
        if len(vector)>max_length:
            if transformer:
                return vector[-max_length:], max_length
            else:
                return vector[:max_length], max_length
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
    
    def unk_mask(self,sentence, length, pad=0, bos=1, eos=2, unk=3):
        unk_sentence = deepcopy(sentence)
        length = deepcopy(length)
        for i, idx in enumerate(sentence):
            if idx != eos:
                unk_sentence[i] = unk
            else:
                break
        return unk_sentence, length
                        
    def process_target(self, target):
        def subprocess(target_1, target_2):
            session_target = []
            reject_target = []
            if target_1 in ["反馈","反馈，结束"]:
                pass
            elif target_1 == "拒绝": # 偏向于拒绝topic # in ["拒绝","拒绝推荐"]:
                if isinstance(target_2, str) and target_2 in self.topic2id:
                    reject_target.append(self.topic2id[target_2])
                elif isinstance(target_2, str) and target_2 in self.entity2id:
                    reject_target.append(self.entity2id[target_2])
                elif isinstance(target_2, list):
                    for t in target_2:
                        if t in self.topic2id:
                            reject_target.append(self.topic2id[t])
                        elif t in self.entity2id:
                            reject_target.append(self.entity2id[t])
            elif target_1 == "拒绝推荐": # 偏向于拒绝 movie
                if isinstance(target_2, str) and target_2 in self.entity2id:
                    reject_target.append(self.entity2id[target_2])
                elif isinstance(target_2, str) and target_2 in self.topic2id:
                    reject_target.append(self.topic2id[target_2])
                elif isinstance(target_2, list):
                    for t in target_2:
                        if t in self.entity2id:
                            reject_target.append(self.entity2id[t])
                        elif t in self.topic2id:
                            reject_target.append(self.topic2id[t])
            elif target_1 == "推荐电影":
                if isinstance(target_2,str) and target_2 in self.entity2id:
                    session_target.append(self.entity2id[target_2])
                elif isinstance(target_2,str) and target_2 in self.topic2id:
                    session_target.append(self.topic2id[target_2])
                elif isinstance(target_2, list):
                    for t in target_2:
                        if t in self.entity2id:
                            session_target.append(self.entity2id[t])
                        elif t in self.topic2id:
                            session_target.append(self.topic2id[t])
            else: # "谈论","请求推荐","允许推荐"
                if isinstance(target_2, str) and target_2 in self.topic2id:
                    session_target.append(self.topic2id[target_2])
                elif isinstance(target_2, str) and target_2 in self.entity2id:
                    session_target.append(self.entity2id[target_2])
                elif isinstance(target_2, list):
                    for t in target_2:
                        if t in self.topic2id:
                            session_target.append(self.topic2id[t])
                        elif t in self.entity2id:
                            session_target.append(self.entity2id[t])
            return session_target, reject_target
        session_target = []
        reject_target = []
        if len(target) == 3:
            session_target, reject_target = subprocess(target[1], target[2])
        if len(target) == 5:
            session_target_12, reject_target_12 = subprocess(target[1], target[2])
            session_target_34, reject_target_34 = subprocess(target[3], target[4])
            session_target = session_target_12 + session_target_34
            reject_target = reject_target_12 + reject_target_34
        session_target = list(set(session_target))
        reject_target = list(set(reject_target))
        return session_target, reject_target


class TGReDial(object):
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

if __name__ == "__main__":
    pass