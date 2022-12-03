import numpy as np
import tqdm
import random
import pickle

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from util.utils import recall_at_k, ndcg_k, get_metric, precision_at_k
from util.utils import Pack
from util.metrics import bleu, distinct, goal_hit, metrics_cal_gen, goal_hit_fuzz

import warnings
warnings.filterwarnings("ignore")

class MIM_Trainer:
    def __init__(self, model, train_dataloader,
                 valid_dataloader,
                 test_dataloader, args):
        self.args = args
        # self.batch_size = args.batch_size
        self.device = args.device
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
    
    def train(self, epoch, logger):
        self.iteration(epoch, self.train_dataloader, logger, train=True)
    
    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)
    
    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))
    
    def iteration(self, epoch, dataloader, logger, train=False, valid=False, test=False):
        if train:
            str_code = "train"
        if valid:
            str_code = "valid"
        if test:
            str_code = "test"
        mim_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="MIM Epoch_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            mim_average_loss = 0.0
            mim_current_loss = 0.0

            for i, data in mim_data_iter:
                outputs, rating_pred = self.model.forward(data)
                loss = 0.0
                assert self.args.use_mim == True
                loss += outputs.mim_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                mim_average_loss += loss.item()
                mim_current_loss = loss.item()

                if (i + 1) % self.args.log_freq_iter == 0:
                    logger.info(", ".join(["Epoch: {}".format(epoch), "MIM loss: {:.4f}".format(mim_average_loss/(i+1)),"Current loss: {:.4f}".format(loss)]))
            
            post_fix = {
                "MIM epoch": epoch,
                "mim_avg_loss": '{:.4f}'.format(mim_average_loss / len(mim_data_iter)),
                "mim_cur_loss": '{:.4f}'.format(mim_current_loss),
            }
            
            if (epoch + 1) % self.args.log_freq == 0:
                logger.info(str(post_fix))
            with open(self.args.total_mim_log, 'a') as f:
                f.write(str(post_fix) + '\n')

class Rec_Trainer:
    def __init__(self, model, train_dataloader,
                 valid_dataloader,
                 test_dataloader, args):
        self.args = args
        # self.batch_size = args.batch_size
        self.device = args.device
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
    
    def train(self, epoch, logger):
        self.iteration(epoch, self.train_dataloader, logger, train=True)

    def valid(self, epoch, logger):
        return self.iteration(epoch, self.valid_dataloader, logger, valid=True)

    def test(self, epoch, logger):
        _, _ = self.iteration(epoch, self.test_dataloader, logger, valid=True)
        return self.iteration(epoch, self.test_dataloader, logger, test=True)
    
    def init_model(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        # print('*'*10, type(state_dict), '*'*10)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self.model)
        print("missing keys:{}".format(missing_keys))
        print('unexpected keys:{}'.format(unexpected_keys))
        print('error msgs:{}'.format(error_msgs))
    
    def get_eval_score(self, epoch, target_list, pred_list, logger):
        recall, ndcg = [], []
        for k in [1, 5, 10, 15, 20, 25]:
            recall.append(recall_at_k(target_list, pred_list, k))
            ndcg.append(ndcg_k(target_list, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.6f}'.format(recall[0]), "NDCG@1": '{:.6f}'.format(ndcg[0]),
            "HIT@5": '{:.6f}'.format(recall[1]), "NDCG@5": '{:.6f}'.format(ndcg[1]),
            "HIT@10": '{:.6f}'.format(recall[2]), "NDCG@10": '{:.6f}'.format(ndcg[2]),
            "HIT@15": '{:.6f}'.format(recall[3]), "NDCG@15": '{:.6f}'.format(ndcg[3]),
            "HIT@20": '{:.6f}'.format(recall[4]), "NDCG@20": '{:.6f}'.format(ndcg[4]),
            "HIT@25": '{:.6f}'.format(recall[5]), "NDCG@25": '{:.6f}'.format(ndcg[5]),
            # "HIT@50": '{:.4f}'.format(recall[6]), "NDCG@50": '{:.4f}'.format(ndcg[6])
        }
        # logger.info(post_fix)
        with open(self.args.total_rec_log, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[1], ndcg[1], recall[2], ndcg[2], recall[4], ndcg[4], recall[5], ndcg[5]], str(post_fix)
    
    def iteration(self, epoch, dataloader, logger, train=False, valid=False, test=False):
        if train:
            str_code = "train"
        if valid:
            str_code = "valid"
        if test:
            str_code = "test"
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation Epoch_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            rec_average_loss = 0.0
            rec_current_loss = 0.0

            mim_loss_avg = 0.0
            rec_loss_avg = 0.0

            for i, data in rec_data_iter:
                outputs, rating_pred = self.model.forward(data)
                loss = 0.0
                if self.args.use_mim:
                    loss += self.args.weight_mim * outputs.mim_loss
                loss += outputs.rec_loss
                self.optimizer.zero_grad()
                loss.backward()
                # self.optimizer.step()
                self.update_params()

                rec_average_loss += loss.item()
                rec_current_loss = loss.item()
                
                if self.args.use_mim:
                    mim_loss_avg += outputs.mim_loss.item()
                rec_loss_avg += outputs.rec_loss.item()

                if self.args.use_mim:
                    if (i + 1) % self.args.log_freq_iter == 0:
                        logger.info(", ".join(["Epoch: {}".format(epoch), "MIM weight: {:.2f}".format(self.args.weight_mim), "MIM loss: {:.4f}".format(mim_loss_avg/(i+1)),"REC loss: {:.4f}".format(rec_loss_avg/(i+1)),"Current loss: {:.4f}".format(loss)]))
                else:
                    if (i + 1) % self.args.log_freq_iter == 0:
                        logger.info(", ".join(["Epoch: {}".format(epoch), "REC loss: {:.4f}".format(rec_loss_avg/(i+1)),"Current loss: {:.4f}".format(loss)]))
            
            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_average_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_current_loss),
                "MIM loss": '{:.4f}'.format(mim_loss_avg / len(rec_data_iter)),
                "REC loss": '{:.4f}'.format(rec_loss_avg / len(rec_data_iter))
            }
            
            if (epoch + 1) % self.args.log_freq == 0:
                logger.info(str(post_fix))
            with open(self.args.total_rec_log, 'a') as f:
                f.write(str(post_fix) + '\n')
        
        if valid:
            self.model.eval()
            with torch.no_grad():
                pred_list = None
                target_list = None
                for i, data in rec_data_iter:
                    outputs, rating_pred = self.model.forward(data)
                    # recommend_output = recommend_output[:, -1, :]
                    _, pred_idx = torch.topk(rating_pred.cpu(), k=50, dim=1)
                    pred_idx = pred_idx.data.numpy()
                    total_target = data.total_target.cpu()
                    batch_size = data.total_target.shape[0]
                    targets = []
                    for bs in range(batch_size):
                        target = total_target[bs][total_target[bs].nonzero().view(-1).tolist()].numpy()
                        targets.append(target)
                    assert len(pred_idx) == len(targets)

                    if i == 0:
                        pred_list = pred_idx
                        target_list = np.array(targets)
                    else:
                        pred_list = np.append(pred_list, pred_idx, axis=0)
                        target_list = np.append(target_list, np.array(targets), axis=0)
                score, post_fix =  self.get_eval_score(epoch, target_list, pred_list, logger)
                logger.info(post_fix)
                return score, post_fix
        
        if test:
            self.model.eval()
            with torch.no_grad():
                total_pred_list = None
                total_target_list = None
                for i, data in rec_data_iter:
                    outputs, rating_pred = self.model.forward(data)
                    _, total_pred_idx = torch.topk(rating_pred.cpu(), k=50, dim=1)

                    total_pred_idx = total_pred_idx.data.numpy()

                    total_target = data.total_target.cpu()

                    batch_size = data.total_target.shape[0]

                    total_targets = []

                    for bs in range(batch_size):
                        tot_target = total_target[bs][total_target[bs].nonzero().view(-1).tolist()].numpy()
                        total_targets.append(tot_target)

                    assert len(total_pred_idx) == len(total_targets)

                    if i == 0:
                        total_pred_list = total_pred_idx
                        total_target_list = np.array(total_targets)

                    else:
                        total_pred_list = np.append(total_pred_list, total_pred_idx, axis=0)
                        total_target_list = np.append(total_target_list, np.array(total_targets), axis=0)

                total_score, total_post = self.get_eval_score(epoch, total_target_list, total_pred_list, logger)
                logger.info("eval total: "+total_post)
                with open(self.args.total_rec_log, 'a') as f:
                    f.write("*"*10+" Eval total: "+str(total_post) + '\n')
                
                return [total_score], ["Eval total: "+total_post]
    
    def update_params(self):
        """
        Perform step of optimization, clipping gradients and adjusting LR
        schedule if needed. Gradient accumulation is also performed if agent
        is called with --update-freq.

        It is recommended (but not forced) that you call this in train_step.
        """
        update_freq = 1
        if update_freq > 1:
            # we're doing gradient accumulation, so we don't only want to step
            # every N updates instead
            self._number_grad_accum = (self._number_grad_accum + 1) % update_freq
            if self._number_grad_accum != 0:
                return

        if self.args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.gradient_clip
            )

        self.optimizer.step()


class Con_Trainer:
    def __init__(self, model, train_dataloader,
                 valid_dataloader,
                 test_dataloader, args):
        self.args = args
        self.device = args.device
        
        self.model = model
        self.id2token = dict([(i,t) for t,i in self.model.token2id.items()])
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)
        
        self.func = lambda x: [y for l in x for y in self.func(l)] if type(x) is list else [x]
        self.EOS = 2
        self.UNK = 3

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
    
    def init_model(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        # print('*'*10, type(state_dict), '*'*10)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self.model)
        print("missing keys:{}".format(missing_keys))
        print('unexpected keys:{}'.format(unexpected_keys))
        print('error msgs:{}'.format(error_msgs))
    
    def vector2sentence(self,batch_sen):
        sentences=[]
        for sen in batch_sen.cpu().numpy().tolist():
            sentence=[]
            for word in sen:
                if word>3:
                    sentence.append(self.id2token[word])
                elif word==3:
                    sentence.append('_UNK_')
            sentences.append(" ".join(sentence))
        return sentences
    
    def num2str(self, number):
        tokens = []
        for x in number:
            if x != self.EOS:
                if x > self.UNK:
                    tokens.append(self.id2token[x])
                elif x == self.UNK:
                    tokens.append("_UNK_")
            else:
                break
        text = " ".join(tokens)
        return text

    def denumericalize(self, numbers):
        if isinstance(numbers, torch.Tensor):
            with torch.cuda.device_of(numbers):
                numbers = numbers.tolist()
        result = []
        for idx in range(len(numbers)):
            num_i = numbers[idx]
            num_i = self.func(num_i)
            str_list = self.num2str(num_i)
            result.append(str_list)
        return result
    
    def write_results(self, results, results_file):
        with open(results_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write("{}\n".format("".join(result)))
        
    def get_eval_score(self, results, preds_all, responses, contexts, test=False):
        context = [result.context.split(" ") for result in results]
        refs = [result.response.split(" ") for result in results]
        hyps = [result.preds.split(" ") for result in results]

        report_message = {}

        avg_len = np.average([len(s) for s in hyps])
        report_message["Avg_Len"] = "{:.3f}".format(avg_len)

        bleu_1, bleu_2 = bleu(hyps, refs)
        report_message["Bleu-1/2"] = "{:.4f}/{:.4f}".format(bleu_1, bleu_2)

        intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(hyps)
        report_message["Inter_Dist-1/2"]="{:.4f}/{:.4f}".format(inter_dist1, inter_dist2)
        report_message["Intra_Dist-1/2"]="{:.4f}/{:.4f}".format(intra_dist1, intra_dist2)
        
        # output_dict_gen = metrics_cal_gen(hyps, refs)
        # report_message["Other-Bleu-1/2/3/4"] = "{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(output_dict_gen["bleu1"],output_dict_gen["bleu2"],output_dict_gen["bleu3"],output_dict_gen["bleu4"])
        # report_message["Other-Dist-1/2/3/4"] = "{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(output_dict_gen["dist1"],output_dict_gen["dist2"],output_dict_gen["dist3"],output_dict_gen["dist4"])
        
        # print("\n".join(report_message))
        if test:
            self.write_results(preds_all, self.args.save_results_file)
            self.write_results(responses, self.args.save_responses_file)
            self.write_results(contexts, self.args.save_contexts_file)
        else:
            self.write_results(preds_all, self.args.valid_save_results_file)
            self.write_results(responses, self.args.valid_save_responses_file)
            self.write_results(contexts, self.args.valid_save_contexts_file)
        
        return [bleu_1, bleu_2, inter_dist1, inter_dist2], report_message
    
    def train(self, epoch, logger):
        self.iteration(epoch, self.train_dataloader, logger, train=True)

    def valid(self, epoch, logger):
        return self.iteration(epoch, self.valid_dataloader, logger, valid=True)

    def test(self, epoch, logger):
        # _, _ = self.iteration(epoch, self.test_dataloader, logger, valid=True)
        return self.iteration(epoch, self.test_dataloader, logger, test=True)
    
    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)
    
    def iteration(self, epoch, dataloader, logger, train=False, valid=False, test=False):
        if train:
            str_code = "train"
        if valid:
            str_code = "valid"
        if test:
            str_code = "test"
        con_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Conversation Epoch_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            
            total_average_loss = 0.0

            mim_loss_avg = 0.0
            rec_loss_avg = 0.0
            con_loss_avg = 0.0
            
            for i, data in con_data_iter:
                outputs, preds = self.model.forward(data, is_training_con=True)
                loss = 0.0
                if self.args.joint_rec:
                    loss += self.args.weight_rec * outputs.rec_loss
                if self.args.joint_mim and self.args.use_mim:
                    loss += self.args.weight_mim * outputs.mim_loss
                loss += outputs.con_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.update_params()
                # self.optimizer.step()
                
                total_average_loss += loss.item()
                
                if self.args.joint_rec:
                    rec_loss_avg += outputs.rec_loss
                if self.args.joint_mim and self.args.use_mim:
                    mim_loss_avg += outputs.mim_loss
                con_loss_avg += outputs.con_loss
                
                if (i+1) % self.args.log_freq_iter == 0:
                    logger.info(", ".join(["Epoch: {}".format(epoch), "MIM weight: {:.2f}".format(self.args.weight_mim), "MIM loss: {:.4f}".format(mim_loss_avg/(i+1)), "REC weight: {:.2f}".format(self.args.weight_rec), "REC loss: {:.4f}".format(rec_loss_avg/(i+1)),"CON Loss: {:.4f}".format(con_loss_avg/(i+1))]))
                    # logger.info(preds[0])
            
            post_fix = {
                "epoch": epoch,
                "total_avg_loss": '{:.4f}'.format(total_average_loss/len(con_data_iter)),
                "MIM loss": '{:.4f}'.format(mim_loss_avg/len(con_data_iter)),
                "REC loss": '{:.4f}'.format(rec_loss_avg/len(con_data_iter)),
                "CON loss": '{:.4f}'.format(con_loss_avg/len(con_data_iter))
            }
            if (epoch + 1) % self.args.log_freq == 0:
                logger.info(str(post_fix))
            with open(self.args.total_con_log, "a") as f:
                f.write(str(post_fix) + "\n")
                
        if valid or test:
            self.model.eval()
            with torch.no_grad():
                generator_outputs = Pack()
                results = []
                responses = []
                contexts = []
                preds_all = []
                for i,data in con_data_iter:
                    outputs, preds = self.model.forward(data, is_training_con=True, is_generator=True)
                    context = self.denumericalize(data.context)
                    response = self.denumericalize(data.response)
                    preds = self.denumericalize(preds)
                    preds_all += preds
                    responses += response
                    contexts += context
                    generator_outputs.add(context=context, response=response, preds=preds)
                    result_batch = generator_outputs.flatten()
                    results += result_batch
                
                if test:
                    scores, post_fix = self.get_eval_score(results, preds_all, responses, contexts, test=True)
                else:
                    scores, post_fix = self.get_eval_score(results, preds_all, responses, contexts)
                logger.info(post_fix)
                return scores, post_fix
    
    def update_params(self):
        """
        Perform step of optimization, clipping gradients and adjusting LR
        schedule if needed. Gradient accumulation is also performed if agent
        is called with --update-freq.

        It is recommended (but not forced) that you call this in train_step.
        """
        update_freq = 1
        if update_freq > 1:
            # we're doing gradient accumulation, so we don't only want to step
            # every N updates instead
            self._number_grad_accum = (self._number_grad_accum + 1) % update_freq
            if self._number_grad_accum != 0:
                return

        if self.args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.gradient_clip
            )

        self.optimizer.step()
        