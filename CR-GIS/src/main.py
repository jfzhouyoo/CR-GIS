'''
 @jfzhou
 main.py
'''

import os
from dataset_tgredial import dataset_tgredial, TGReDial
from dataset_opendialkg import Opendial, dataset_opendialkg, OpenDialKG
from util.utils import set_random_seed, get_logger, Pack, list2tensor, str2bool
from util.utils import EarlyStopping
from CRGIS import CRGIS
from trainers import MIM_Trainer, Rec_Trainer, Con_Trainer

from copy import deepcopy
import numpy as np
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import argparse
import pickle
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

# def setup_args():
#     return parser

class CRSDialLoader(Dataset):
    def __init__(self, dataset):
        self.dataset = deepcopy(dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
    
    @staticmethod
    def collate_fn(device=-1):
        def collate(data_list):
            batch = Pack()
            keys = ["user", "conv_id", "local_id", "sample_id", "context", "context_length", "response", "response_length", "mask_response", "mask_response_length", "target_history", "entities_history", "total_entities_history", "total_entities_history_mask", "movie_target", "topic_target", "total_target", "negative_target", "recommender_rec_movie", "recommender_rec_topic", "recommender_rec", "rec_movie", "rec_topic", "rec", "context_uttr_list", "context_uttr_num", "context_uttr_entity_list", "context_uttr_entity_num","entities_history_vector", "adjacent_matrix", "alias_data", "alias_context_uttr_entity_list"]
            for i, key in enumerate(keys):
                if "rec" not in key:
                    batch[key] = list2tensor([x[i] for x in data_list])
                else:
                    batch[key] = list2tensor([x[i] for x in data_list])
                # batch[key] = list2tensor([x[i] for x in data_list])

            if device.type != "cpu":
                batch = batch.cuda(device=device.index)
            return batch
        return collate

    def create_adjacent_matrix(self):
        print("*"*10+" create adjacent matrix...... "+"*"*10)
        num_entity = []
        for data in self.dataset:
            num_entity.append(len(np.unique(data[12]))) # total_entities_history
        max_n_entity = np.max(num_entity)

        for data in tqdm(self.dataset):
            total_entities_history = data[12]
            entities_history = np.unique(total_entities_history) # total_entities_history, include id:0
            entities_history_vector = entities_history.tolist()+(max_n_entity-len(entities_history))*[0] # padding
            data.append(np.array(entities_history_vector)) # Note that a element is added into data
            adjacent_matrix = np.zeros((max_n_entity, max_n_entity))
            for i in np.arange(len(total_entities_history)-1):
                if total_entities_history[i+1] == 0:
                    break
                u = np.where(entities_history == total_entities_history[i])[0][0] # find the index of data[12][i] in entities_history
                v = np.where(entities_history == total_entities_history[i+1])[0][0]
                adjacent_matrix[u][v] = 1
            sum_in = np.sum(adjacent_matrix, 0)
            sum_in[np.where(sum_in==0)] = 1
            adjacent_in = np.divide(adjacent_matrix, sum_in)
            sum_out = np.sum(adjacent_matrix, 1)
            sum_out[np.where(sum_out==0)] = 1
            adjacent_out = np.divide(adjacent_matrix.transpose(), sum_out)
            adjacent_matrix = np.concatenate([adjacent_in, adjacent_out]).transpose()
            # update into data
            data.append(adjacent_matrix)
            data.append(np.array([np.where(entities_history==i)[0][0] for i in total_entities_history])) # alias data, align index
            # data = tuple(data)
            context_uttr_entity_list = data[26]
            data.append(np.array([[np.where(entities_history==i)[0][0] for i in context_uttr_entity] for context_uttr_entity in context_uttr_entity_list]))
        print("*"*10+" create adjacent matrix done! "+"*"*10)

    def create_batches(self, batch_size, shuffle=False, device=-1):
        """
        create_batches
        """
        self.create_adjacent_matrix()
        loader = DataLoader(dataset=self,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=self.collate_fn(device),
                            pin_memory=False)
        return loader

def main(args):

    if args.dataset == "tgredial":
        dataset = dataset_tgredial
        loader = TGReDial
    elif args.dataset == "opendialkg":
        dataset = dataset_opendialkg
        loader = OpenDialKG
    else:
        print("Dataset Error!!! Pleace choose one dataset from [tgredial, opendialkg]!")

    # train_data loader
    train_dataloader = CRSDialLoader(loader(dataset(args.train_data, vars(args)).data_process()).dataset2list(align_context=args.max_entity_length, align_response=args.max_target_length)).create_batches(args.batch_size, shuffle=True, device=args.device)

    # valid_data loader
    valid_dataloader = CRSDialLoader(loader(dataset(args.valid_data, vars(args)).data_process()).dataset2list(align_context=args.max_entity_length, align_response=args.max_target_length)).create_batches(args.batch_size, shuffle=False, device=args.device)
    # train_dataloader = valid_dataloader
    # test_dataloader = valid_dataloader

    # test_data loader
    test_dataloader = CRSDialLoader(loader(dataset(args.test_data, vars(args)).data_process()).dataset2list(align_context=args.max_entity_length, align_response=args.max_target_length)).create_batches(args.batch_size, shuffle=False, device=args.device)

    model = CRGIS(args).to(args.device)
    
    # MIM Trainer
    mim_trainer = MIM_Trainer(model, train_dataloader, 
                                     valid_dataloader, 
                                     test_dataloader, args)
    mim_checkpoint_path = args.save_dir + "mim_model.ckpt"
    # mim_logger = get_logger(args.mim_log)
    
    # REC Trainer
    rec_trainer = Rec_Trainer(model, train_dataloader, 
                                     valid_dataloader, 
                                     test_dataloader, args)
    rec_checkpoint_path = args.save_dir + "rec_model.ckpt"
    # rec_logger = get_logger(args.rec_log)

    # CON Trainer
    con_trainer = Con_Trainer(model, train_dataloader,
                                     valid_dataloader,
                                     test_dataloader, args)
    con_checkpoint_path = args.save_dir + "con_model.ckpt"
    # con_logger = get_logger(args.con_log)

    if args.use_mim and args.is_pretrain_mim:
        mim_logger = get_logger(args.mim_log)
        mim_logger.info("MIM Pretraining Start...")
        # mim_trainer.model.apply(mim_trainer.model.init_weights)
        mim_trainer.model.word_embedings = mim_trainer.model._create_embeddings(args, mim_trainer.model.token2id, args.encoder_token_emb_dim, mim_trainer.model.padding_idx) # 确保word_embedding不会因初始化参数而改变
        mim_logger.info(args)
        for epoch in range(args.mim_epochs):
            mim_trainer.train(epoch, mim_logger)
            mim_trainer.save(mim_checkpoint_path)
    
    if args.is_training_rec:
        rec_logger = get_logger(args.rec_log)
        if  args.use_mim and args.is_load_mim and args.is_load_rec==False:
            rec_logger.info("Load the MIM model from {}".format(mim_checkpoint_path))
            rec_trainer.model.load_state_dict(torch.load(mim_checkpoint_path))
        if args.is_load_rec:
            rec_logger.info("Load the REC model from {}".format(rec_checkpoint_path))
            rec_trainer.model.load_state_dict(torch.load(rec_checkpoint_path))
        rec_logger.info("Recommender Training Start...")
        rec_logger.info(args)
        early_stopping = EarlyStopping(rec_checkpoint_path, patience=10, verbose=True)
        for epoch in range(args.rec_epochs):
            rec_trainer.train(epoch, rec_logger)
            scores, _ = rec_trainer.valid(epoch, rec_logger)
            early_stopping(np.array([scores[-2], scores[-1]]), rec_trainer.model)
            if early_stopping.early_stop:
                rec_logger.info("Early stopping")
                break
        rec_logger.info("Load the best model to eval test data...")
        rec_trainer.model.load_state_dict(torch.load(rec_checkpoint_path))
        scores, result_info = rec_trainer.test(0, rec_logger)

        # rec_logger.info(result_info)
        with open(args.total_rec_log, 'a') as f:
            f.write("\n".join(result_info) + '\n')
    
    if args.eval_rec:
        # if args.is_load_con:
        #     rec_logger = get_logger(args.eval_rec_after_con_log)
        #     rec_trainer.model.load_state_dict(torch.load(con_checkpoint_path))
        #     rec_logger.info('Load CON model from {} for test!'.format(con_checkpoint_path))
        # else:
        rec_logger = get_logger(args.eval_rec_log)
        rec_trainer.model.load_state_dict(torch.load(rec_checkpoint_path))
        rec_logger.info('Load REC model from {} for test!'.format(rec_checkpoint_path))
        scores, result_info = rec_trainer.test(0, rec_logger)

    if args.is_training_con:
        con_logger = get_logger(args.con_log)
        con_logger.info("Load the REC model from {}".format(rec_checkpoint_path))
        # con_trainer.model.load_state_dict(torch.load(rec_checkpoint_path)) # 加载上面预训练的推荐模型
        con_trainer.init_model(rec_checkpoint_path)
        if args.is_load_con:
            con_logger.info("Load the CON model from {}".format(con_checkpoint_path))
            con_trainer.init_model(con_checkpoint_path)
        con_logger.info("Conversation Training Start...")
        con_logger.info(args)
        early_stopping = EarlyStopping(con_checkpoint_path, patience=10, verbose=True)
        for epoch in range(args.con_epochs):
            con_trainer.train(epoch, con_logger)
            scores, preds = con_trainer.valid(epoch, con_logger)
            # early_stopping(np.array([scores[1], scores[-1]]), con_trainer.model)
            # if early_stopping.early_stop:
            #     con_logger.info("Early stopping")
            #     break
            con_trainer.save(con_checkpoint_path)
        con_logger.info("Load the best model to eval test data...")
        con_trainer.model.load_state_dict(torch.load(con_checkpoint_path))
        scores, result_info = con_trainer.test(0, con_logger)
        con_logger.info(result_info)
        with open(args.total_con_log, 'a') as f:
            f.write("\n".join(result_info) + '\n')
    
    if args.is_generator:
        con_logger = get_logger(args.eval_con_log)
        con_trainer.model.load_state_dict(torch.load(con_checkpoint_path))
        con_logger.info('Load model from {} for test!'.format(con_checkpoint_path))
        scores, result_info = con_trainer.test(0, con_logger)
        con_logger.info(result_info)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset","--dataset",type=str,default="tgredial")

    parser.add_argument("-train_data","--train_data",type=str,default="train_dials.json")
    parser.add_argument("-valid_data","--valid_data",type=str,default="valid_dials.json")
    parser.add_argument("-test_data","--test_data",type=str,default="test_dials.json")
    parser.add_argument("-kg","--kg",type=str,default="kg.pkl")
    parser.add_argument("-word2vec","--word2vec",type=str,default="word2vec.npy")

    # model args
    parser.add_argument("-max_context_length", "--max_context_length", type=int, default=128)
    parser.add_argument("-max_response_length", "--max_response_length", type=int, default=50) # 50
    parser.add_argument("-history_window_size", "--history_window_size", type=int, default=10)
    parser.add_argument("-max_uttr_length", "--max_uttr_length", type=int, default=50) # 64
    parser.add_argument("-max_uttr_entity_num", "--max_uttr_entity_num", type=int, default=10)
    
    parser.add_argument("-max_entity_num", "--max_entity_num", type=int, default=15)
    parser.add_argument("-max_entity_length", "--max_entity_length", type=int, default=50)
    parser.add_argument("-max_target_length", "--max_target_length", type=int, default=50)
    
    parser.add_argument("-entity_num", "--entity_num", type=int, default=63312) # include padding 0
    parser.add_argument("-relation_num", "--relation_num", type=int, default=61)
    parser.add_argument("-movie_num", "--movie_num", type=int, default=33531) # total 60742
    parser.add_argument("-topic_num", "--topic_num", type=int, default=60742)
    
    parser.add_argument("-opendialkg_entity_num", "--opendialkg_entity_num", type=int, default=100925) # include padding 0
    parser.add_argument("-opendialkg_relation_num", "--opendialkg_relation_num", type=int, default=1382)
    parser.add_argument("-opendialkg_movie_num", "--opendialkg_movie_num", type=int, default=100925) # total 60742
    parser.add_argument("-opendialkg_topic_num", "--opendialkg_topic_num", type=int, default=100925)
    
    parser.add_argument("-sgnn_layers", "--sgnn_layers", type=int, default=6)
    parser.add_argument("-num_attention_layers", "--num_attention_layers", type=int, default=2)
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument("-num_bases", "--num_bases", type=int, default=8)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    
    # train args
    parser.add_argument("-batch_size", "--batch_size", type=int, default=32)
    parser.add_argument("-lr", "--lr", type=float, default=0.001)
    parser.add_argument("-mim_epochs", "--mim_epochs", type=int, default=30)
    parser.add_argument("-rec_epochs", "--rec_epochs", type=int, default=30)
    parser.add_argument("-con_epochs", "--con_epochs", type=int, default=1)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("-embedding_dim", "--embedding_dim", type=int, default=128)
    parser.add_argument("-hidden_size", "--hidden_size", type=int, default=128)
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--weight_mim", type=float, default=0.5)
    parser.add_argument("-joint_rec", "--joint_rec", type=str2bool, default=True)
    parser.add_argument("-joint_mim", "--joint_mim", type=str2bool, default=False)
    parser.add_argument("--weight_rec", type=float, default=0.2)
    parser.add_argument("-train_rec_only", "--train_rec_only", type=str2bool, default=False)

    # hier encoder
    parser.add_argument("-encoder_token_emb_dim", "--encoder_token_emb_dim", type=int, default=300)
    parser.add_argument("-encoder_uttr_emb_dim", "--encoder_uttr_emb_dim", type=int, default=300) # 512
    parser.add_argument("-encoder_hidden_size", "--encoder_hidden_size", type=int, default=300) # 512
    parser.add_argument("-encoder_num_layers", "--encoder_num_layers", type=int, default=4)
    parser.add_argument("-encoder_num_head", "--encoder_num_head", type=int, default=2)
    parser.add_argument("-head_dim", "--head_dim", type=int, default=300) # 512
    parser.add_argument("-token_max_len", "--token_max_len", type=int, default=50)
    parser.add_argument("-uttr_max_len", "--uttr_max_len", type=int, default=10)
    parser.add_argument("-encoder_dropout", "--encoder_dropout", type=float, default=0.1)
    
    # decoder
    parser.add_argument("-decoder_num_layers", "--decoder_num_layers", type=int, default=2)
    parser.add_argument("-decoder_hidden_size", "--decoder_hidden_size", type=int, default=300)
    parser.add_argument("-decoder_num_head", "--decoder_num_head", type=int, default=2)
    parser.add_argument("-ffn_size","--ffn_size",type=int,default=300) # 512
    parser.add_argument("-decoder_dropout", "--decoder_dropout", type=float, default=0.1)
    parser.add_argument("-decoder_attention_dropout","--decoder_attention_dropout",type=float,default=0.0)
    parser.add_argument("-decoder_relu_dropout","--decoder_relu_dropout",type=float,default=0.1)
    parser.add_argument("-learn_positional_embeddings","--learn_positional_embeddings",type=bool,default=False)
    parser.add_argument("-embeddings_scale","--embeddings_scale",type=bool,default=True)
    parser.add_argument("-gradient_clip","--gradient_clip",type=float,default=0.1)
    parser.add_argument("--force_copy", type=str2bool, default=True)


    parser.add_argument("-is_pretrain_mim", "--is_pretrain_mim", type=str2bool, default=False)  
    parser.add_argument("-is_training_rec", "--is_training_rec", type=str2bool, default=False)
    parser.add_argument("-is_load_mim", "--is_load_mim", type=str2bool, default=False)
    parser.add_argument("-is_load_rec", "--is_load_rec", type=str2bool, default=False)
    parser.add_argument("-eval_rec", "--eval_rec", type=str2bool, default=False)
    parser.add_argument("-is_training_con", "--is_training_con", type=str2bool, default=False)
    parser.add_argument("-freeze_rec", "--freeze_rec", type=str2bool, default=False)
    parser.add_argument("-is_generator", "--is_generator", type=str2bool, default=False)
    parser.add_argument("-is_load_con", "--is_load_con", type=str2bool, default=False)
    parser.add_argument('--gpu', type=str, default="2", help='gpu device.')
    parser.add_argument("-seed", "--seed", type=int, default=1234)

    parser.add_argument("-use_mim", "--use_mim", type=str2bool, default=True)

    # save
    parser.add_argument("-save_dir","--save_dir",type=str,default="../models/")
    
    # log
    parser.add_argument("-mim_log","--mim_log",type=str,default="../logs/mim_log.log")
    parser.add_argument("-total_mim_log","--total_mim_log",type=str,default="../logs/total_mim_log.txt")
    
    parser.add_argument("-rec_log","--rec_log",type=str,default="../logs/rec_log.log")
    parser.add_argument("-eval_rec_log","--eval_rec_log",type=str,default="../logs/eval_rec_log.log")
    parser.add_argument("-eval_rec_after_con_log","--eval_rec_after_con_log",type=str,default="../logs/eval_rec_after_con_log.log")
    parser.add_argument("-total_rec_log","--total_rec_log",type=str,default="../logs/total_rec_log.txt")
        
    parser.add_argument("-con_log","--con_log",type=str,default="../logs/con_log.log")
    parser.add_argument("-eval_con_log","--eval_con_log",type=str,default="../logs/eval_con_log.log")
    parser.add_argument("-total_con_log","--total_con_log",type=str,default="../logs/total_con_log.txt")
    parser.add_argument("-log_freq", "--log_freq", type=int, default=1)
    parser.add_argument("-log_freq_iter", "--log_freq_iter", type=int, default=500)
    
    parser.add_argument("-save_results_file","--save_results_file",type=str,default="../models/results.txt")
    parser.add_argument("-save_responses_file","--save_responses_file",type=str,default="../models/responses.txt")
    parser.add_argument("-save_contexts_file","--save_contexts_file",type=str,default="../models/contexts.txt")
    
    parser.add_argument("-valid_save_results_file","--valid_save_results_file",type=str,default="../models/valid_results.txt")
    parser.add_argument("-valid_save_responses_file","--valid_save_responses_file",type=str,default="../models/valid_responses.txt")
    parser.add_argument("-valid_save_contexts_file","--valid_save_contexts_file",type=str,default="../models/valid_contexts.txt")
    
    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else 'cpu'

    if args.dataset == "opendialkg":
        args.entity_num = args.opendialkg_entity_num
        args.relation_num = args.opendialkg_relation_num
        args.movie_num = args.opendialkg_movie_num
        args.topic_num = args.opendialkg_topic_num

    set_random_seed(args.seed)

    main(args)