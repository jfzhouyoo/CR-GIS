'''
 @jfzhou
 main model
'''

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

from util.utils import _create_entity_embeddings, Pack
from util.discriminator import Discriminator
from util.attention import SelfAttentionLayer
from util.modules import Encoder, LayerNorm
from util.criterions import NLLLoss, CopyGeneratorLoss
from util.metrics import accuracy
from sgnn import SGNN
from star_transformer import StarTransEnc as Trans_Encoder
from vanilla_transformer_decoder import _build_decoder as Trans_Decoder
from vanilla_transformer_encoder import TransformerEncoderLayer as Encoder_layer

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, *input):
        raise NotImplementedError

    def __repr__(self):
        main_string = super(BaseModel, self).__repr__()
        num_parameters = sum([p.nelement() for p in self.parameters()])
        main_string += "\nNumber of parameters: {}\n".format(num_parameters)
        return main_string

    def save(self, filename):
        torch.save(self.state_dict(), filename)
        print("Saved model state to '{}'!".format(filename))

    def load(self, filename):
        """
        load
        """
        if os.path.isfile(filename):
            state_dict = torch.load(
                filename, map_location=lambda storage, loc: storage)
            self.load_state_dict(state_dict, strict=False)
            print("Loaded model state from '{}'".format(filename))
        else:
            print("Invalid model state file: '{}'".format(filename))


class CRGIS(BaseModel):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.entity_num = args.entity_num
        self.relation_num = args.relation_num
        self.num_bases = args.num_bases
        self.max_entity_length = args.max_entity_length
        self.max_response_length = args.max_response_length
        self.register_buffer('TARGET', torch.LongTensor([self.entity_num+1]))
        self.bos = 1
        self.eos = 2
        self.unk = 3
        self.cls = 4
        self.register_buffer('START', torch.LongTensor([self.bos]))
        self.sgnn_layers = args.sgnn_layers
        self.lr = args.lr
        # self.batch_size = args.batch_size
        self.device = args.device
        self.use_mim = args.use_mim
        self.force_copy = args.force_copy

        self.embedding_dim = args.embedding_dim # entity
        self.kg = pickle.load(open(f"../data/{args.dataset}/{args.kg}", "rb"))
        self.entity_padding = 0
        # self.entity_edge_sets = self._edge_list4GCN()
        # self.GCN = GCNConv(self.embedding_dim, self.embedding_dim)
        self.entity_embeddings = _create_entity_embeddings(self.entity_num+1, self.embedding_dim, self.entity_padding)
        if args.dataset == "tgredial":
            self.RGCN = RGCNConv(self.entity_num+1, self.embedding_dim, self.relation_num+1, num_bases=self.num_bases)
            self.edge_idx, self.edge_type = self._edge_list4TgReDialRGCN()
        elif args.dataset == "opendialkg":
            edge_list, self.relation_num = self._edge_list4OpenDialKGRGCN(self.kg)
            edge_list = list(set(edge_list))
            self.edge_list=torch.LongTensor(edge_list).to(args.device)
            self.edge_idx = self.edge_list[:, :2].t()
            self.edge_type = self.edge_list[:, 2]
            self.RGCN = RGCNConv(self.entity_num+1, self.embedding_dim, self.relation_num, num_bases=self.num_bases)

        self.SGNN = SGNN(self.embedding_dim, self.sgnn_layers, self.device)
        self.dicsriminator = Discriminator(self.embedding_dim)
        self.mim_norm = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.self_attn = SelfAttentionLayer(self.embedding_dim, self.embedding_dim)
        self.position_embeddings = nn.Embedding(args.max_entity_length, args.hidden_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        # self.entity_encoder = Encoder(args)
        self.entity_encoder = Encoder_layer(args.hidden_size, args.num_attention_heads, args.hidden_size, args.hidden_dropout_prob)
        self.output_en = nn.Linear(self.embedding_dim, self.entity_num+1)
        
        # hier encoder Star Transformer
        # self.train_dialog = args.train_dialog
        # Load word2vec
        self.padding_idx = 0
        if args.dataset == "tgredial":
            self.token2id = pickle.load(open(f"../data/tgredial/word2id.pkl", "rb"))
        elif args.dataset == "opendialkg":
            self.token2id = json.load(open("../data/opendialkg/word2index.json",encoding="utf-8"))
            self.token2id["pad"] = 0
            self.token2id["bos"] = 1
            self.token2id["eos"] = 2
            self.token2id["unk"] = 3
            self.token2id["cls"] = 4
        self.token2id["__split__"] = len(self.token2id)
        self.vocab_size = len(self.token2id)
        self.word_embedings = self._create_embeddings(args, self.token2id, args.encoder_token_emb_dim, self.padding_idx)
        self.word_embedings_dim = args.encoder_token_emb_dim
        # Encoder
        self.token_encoder = Trans_Encoder(args.encoder_token_emb_dim, args.encoder_hidden_size, args.encoder_num_layers, args.encoder_num_head, args.head_dim, args.token_max_len, dropout=args.encoder_dropout, mode="token", embeddings=self.word_embedings)
        self.token_merge_fc = nn.Sequential(nn.Dropout(args.encoder_dropout),
                                            nn.Linear(args.encoder_hidden_size*2, args.encoder_hidden_size),
                                            LayerNorm(args.encoder_hidden_size, eps=1e-12))
        
        self.token_encoder_layer = Encoder_layer(args.encoder_hidden_size, args.encoder_num_head, args.ffn_size, args.encoder_dropout)
        
        self.uttr_encoder = Trans_Encoder(args.encoder_uttr_emb_dim, args.encoder_hidden_size, args.encoder_num_layers, args.encoder_num_head, args.head_dim, args.uttr_max_len, dropout=args.encoder_dropout, mode="uttr")
        self.uttr_merge_fc = nn.Sequential(nn.Dropout(args.encoder_dropout),
                                           nn.Linear(args.encoder_hidden_size*2, args.encoder_hidden_size),
                                           LayerNorm(args.encoder_hidden_size, eps=1e-12))
        self.entity_node_fc = nn.Linear(args.embedding_dim, args.encoder_hidden_size)
        self.entity_relay_fc = nn.Linear(args.embedding_dim, args.encoder_hidden_size)
        # decoder Transformer
        self.decoder = Trans_Decoder(vars(args), self.token2id, self.word_embedings, self.padding_idx)
        self.user_norm = nn.Linear(args.embedding_dim, args.encoder_hidden_size)
        
        # Loss
        self.CELoss = nn.CrossEntropyLoss(reduce=False)
        self.BCELoss = nn.BCELoss(reduction="none")
        
        if self.padding_idx is not None:
            self.weight = torch.ones(len(self.token2id))
            self.weight[self.padding_idx] = 0
        else:
            self.weight = 0
        
        self.weight = self.weight.to(self.device)

        self.nll_loss = NLLLoss(weight=self.weight, ignore_index=self.padding_idx,
                                reduction='mean')
        self.copy_gen_loss = CopyGeneratorLoss(vocab_size=self.vocab_size,
                                               force_copy=self.force_copy,
                                               unk_index=self.unk,
                                               ignore_index=self.padding_idx)
          
        # self.apply(self.init_weights)
        if args.is_training_con and args.freeze_rec:
            params = [self.RGCN.parameters(), self.SGNN.parameters(),self.dicsriminator.parameters(), self.mim_norm.parameters(), self.self_attn.parameters(), self.position_embeddings.parameters(), self.LayerNorm.parameters(), self.entity_encoder.parameters(), self.output_en.parameters()]
            for param in params:
                for pa in param:
                    pa.requires_grad = False
    
    def _create_embeddings(self, args, dictionary, embedding_size, padding_idx):
        """Create and initialize word embeddings."""
        word_embeddings = nn.Embedding(len(dictionary), embedding_size, padding_idx)
        word_embeddings.weight.data.copy_(torch.from_numpy(np.load(f"../data/{args.dataset}/{args.word2vec}")))
        return word_embeddings
    
    def _edge_list4GCN(self):
        edges = set()
        for head in self.kg:
            for entity_type, relation, tail in self.kg[head]:
                edges.add((head, tail))
                edges.add((tail, head))
        edge_set = [[co[0] for co in list(edges)], 
                    [co[1] for co in list(edges)]]
        return torch.LongTensor(edge_set).cuda()
    
    def _edge_list4TgReDialRGCN(self):
        '''if the performence of GCN is bad, then use RGCN'''
        self_loop = self.relation_num + 1
        edges = set()
        for head in self.kg:
            edges.add((head, head, self_loop))
            for entity_type, relation, tail in self.kg[head]:
                if head != tail:
                    edges.add((head, tail, relation))
                    edges.add((tail, head, relation))
        # edge_sets = torch.as_tensor(edges, dtype=torch.long)
        relation_idx = {}
        for head, tail, relation in edges:
            if relation not in relation_idx:
                relation_idx[relation] = len(relation_idx)
        edge_sets = [(head, tail, relation_idx[relation]) for head, tail, relation in edges]
        # edge_sets = [edge for edge in edges]
        edge_sets = torch.LongTensor(edge_sets).cuda()
        edge_idx = edge_sets[:, :2].t()
        edge_type = edge_sets[:, 2]
        return edge_idx.to(self.device), edge_type.to(self.device)
    
    def _edge_list4OpenDialKGRGCN(self, kg):
        edge_list = []
        pair_dict = {}
        for entity in kg:
            for tail_and_relation in kg[entity]:
                if entity == tail_and_relation[1]:
                    edge_list.append((entity, entity, tail_and_relation[0]))
                else:
                    if (entity,tail_and_relation[1]) not in pair_dict:
                        edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                        pair_dict[(entity,tail_and_relation[1])] = tail_and_relation[0]
                    else:
                        edge_list.append((tail_and_relation[1], entity, pair_dict[(entity,tail_and_relation[1])]))
                    # edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))
        
        relation_cnt = defaultdict(int)
        relation_idx = {}
        for h, t, r in edge_list:
            relation_cnt[r] += 1
        for h, t, r in edge_list:
            if relation_cnt[r] > 1000 and r not in relation_idx:
                relation_idx[r] = len(relation_idx)
        return [(h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > 1000], len(relation_idx)

    def _target(self, bsz):
        """Return bsz target index."""
        return self.TARGET.detach().expand(bsz, 1)
    
    def _starts(self, bsz):
        """Return bsz start tokens."""
        return self.START.detach().expand(bsz, 1)
    
    def compute_mim_loss(self, star_hidden, pos_hidden, neg_hidden, pos_mask):
        seq_num = pos_hidden.shape[1]
        embed_size = pos_hidden.shape[-1]
        star_hidden = star_hidden.unsqueeze(1).repeat(1, seq_num, 1).view([-1, embed_size])
        mim_positive_score = self.dicsriminator(star_hidden, pos_hidden.view([-1, embed_size]))
        mim_positive_score = mim_positive_score.squeeze()*pos_mask.flatten()
        mim_negative_score = self.dicsriminator(star_hidden, neg_hidden.view([-1, embed_size]))
        mim_negative_score = mim_negative_score.squeeze()*pos_mask.flatten()
        mim_logits = torch.cat((mim_positive_score, mim_negative_score))
        mim_labels = torch.cat((torch.ones_like(mim_positive_score), torch.zeros_like(mim_negative_score)))
        mim_loss = self.BCELoss(mim_logits, mim_labels)
        return mim_loss
    
    def informax_score(self, star_hidden, target_feature):
        '''
        :param star_hidden: [batch_size, embedding_dim]
        :param target_feature [batch_size, target_num, embedding_dim]
        :param score [batch_size*target_num]
        '''
        target_num = target_feature.size(1)
        star_hidden = self.mim_norm(star_hidden.unsqueeze(1).repeat(1, target_num, 1).view([-1, self.embedding_dim]))
        # star_hidden = star_hidden.unsqueeze(1).repeat(1, target_num, 1).view([-1, self.embedding_dim])
        target_feature = target_feature.view([-1, self.embedding_dim])
        score = torch.mul(star_hidden, target_feature)
        return torch.sigmoid(torch.sum(score, -1))

    
    def add_position_embedding(self, sequence, seq_embeddings):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = seq_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        return sequence_emb
    
    def cross_entropy(self, seq_out, pos_ids, pos_emb, neg_ids, neg_emb):
        pos = pos_emb.view(-1, pos_emb.size(2)) # [batch_size*seq_len, embedding_dim]
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.embedding_dim) # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.max_entity_length).float() # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss
    
    def encoder(self, data, entities_history_hidden, star_node_hidden):
        
        context_uttr = data.context_uttr_list
        context_uttr_num = data.context_uttr_num
        context_uttr = torch.cat([uttr[:uttr_num] for uttr, uttr_num in zip(context_uttr.cpu(), context_uttr_num.cpu())], dim=0).to(self.device)
        mask_context_uttr = torch.sign(context_uttr)
        assert context_uttr.size(0) == torch.sum(context_uttr_num)
        
        context_uttr_entity = data.context_uttr_entity_list
        context_uttr_entity_num = data.context_uttr_entity_num
        mask_context_uttr_entity = torch.sign(context_uttr_entity)
        
        # token encoder
        token_nodes_out, token_relay_out = self.token_encoder(context_uttr, mask_context_uttr)
        token_out = self.token_merge_fc(torch.cat([token_nodes_out, token_relay_out.unsqueeze(1).repeat(1,token_nodes_out.size(1),1)], -1)) 
        token_out = self.token_encoder_layer(token_out, ~mask_context_uttr.bool())
        
        # obtain CLS representation
        sent_list = torch.split(token_out[:,0,:], context_uttr_num.cpu().numpy().tolist())
        sent_hid = pad_sequence(sent_list, batch_first=True, padding_value=0.) 
        sent_mask_list = [mask_context_uttr.new_ones([length]) for length in context_uttr_num]
        sent_mask = pad_sequence(sent_mask_list, batch_first=True, padding_value=0)
        
        # uttr encoder
        # obtain the feature of entities which are in uttr
        get = lambda i: entities_history_hidden[i][data.alias_context_uttr_entity_list[i]]  # look up
        context_uttr_entity_embeds = torch.stack([get(i) for i in torch.arange(len(data.alias_context_uttr_entity_list)).long()]) # from recommendation's star graph shape: [batch, uttr_length, entitiy_num, hidden_size]
        context_uttr_entity_embeds_list = [embeds[:uttr_num] for embeds, uttr_num in zip(context_uttr_entity_embeds.cpu(), context_uttr_num.cpu())]
        context_uttr_entity_embeds = pad_sequence(context_uttr_entity_embeds_list, batch_first=True, padding_value=0.)
        context_uttr_entity_embeds_list = [pad_sequence([emb[:num] for emb, num in zip(embeds, entity_num)],batch_first=True, padding_value=0.).transpose(0,1) for embeds, entity_num in zip(context_uttr_entity_embeds.cpu(), context_uttr_entity_num.cpu())]
        context_uttr_entity_embeds = pad_sequence(context_uttr_entity_embeds_list, batch_first=True, padding_value=0.).transpose(1,2).to(self.device)
        assert context_uttr_entity_embeds.size(1)==sent_hid.size(1)
        context_uttr_entity_embeds = self.entity_node_fc(context_uttr_entity_embeds) # [batch_size,uttr_length, entity_num, hidden_size]
        context_star_node_embeds = self.entity_relay_fc(star_node_hidden) # [batch_size, hidden_size]
        
        context_uttr_entity_mask = mask_context_uttr_entity[:,:context_uttr_entity_embeds.size(1),:context_uttr_entity_embeds.size(2)]
        D = context_uttr_entity_embeds.size(-1)
        context_uttr_entity_mask = context_uttr_entity_mask.unsqueeze(-1).repeat(1,1,1,D).eq(False)
        context_uttr_entity_embeds = context_uttr_entity_embeds.masked_fill_(context_uttr_entity_mask,0)
        
        uttr_nodes_out, uttr_relay_out = self.uttr_encoder(sent_hid, sent_mask, context_uttr_entity_embeds, context_star_node_embeds)
        uttr_out = self.uttr_merge_fc(torch.cat([uttr_nodes_out, uttr_relay_out.unsqueeze(1).repeat(1,uttr_nodes_out.size(1), 1)], -1)) 
        
        uttr_mask_list = [mask_context_uttr.new_ones([length]) for length in context_uttr_num]
        uttr_mask = pad_sequence(uttr_mask_list, batch_first=True, padding_value=0)
             
        return uttr_out, uttr_mask.bool()

    def decode_forced(self, data, encoder_states, user_embed, outputs):
        tgt_tokens = data.response
        bsz = tgt_tokens.size(0)
        seq_length = tgt_tokens.size(1)
        inputs = tgt_tokens.narrow(1, 0, seq_length-1)
        inputs = torch.cat([self._starts(bsz), inputs], 1)
        user_embed_norm = self.user_norm(user_embed)
        latent, _ = self.decoder(inputs, encoder_states, user_embed_norm)
        # if latent.size(-1) != self.word_embedings_dim:
        #     latent = self.output_layer(latent)
        # latent = self.output_layer(torch.cat([latent, user_embed_norm.unsqueeze(1).repeat(1, seq_length, 1)], -1))
        logits = F.linear(latent, self.word_embedings.weight)
        _, preds = logits.max(dim=2)
        con_loss = torch.mean(self.gen_cross_entropy_loss(logits, tgt_tokens))
        outputs.add(con_loss=con_loss)
        return outputs, logits, preds
    
    def gen_cross_entropy_loss(self, output, scores):
        score_view = scores.view(-1)
        output_view = output.view(-1, output.size(-1))
        loss = self.CELoss(output_view.cuda(), score_view.cuda())
        return loss
    
    def gen_nll_loss(self, logits, tgt_tokens, unk_mask_target, outputs):
        nll_loss_ori = self.copy_gen_loss(scores=logits.transpose(1, 2).contiguous(), align=unk_mask_target, target=tgt_tokens)
        nll_loss = torch.mean(torch.sum(nll_loss_ori, dim=-1))
        num_words = tgt_tokens.ne(self.padding_idx).sum()  # .item()
        ppl = nll_loss_ori.sum() / num_words
        ppl = ppl.exp()
        acc = accuracy(logits, tgt_tokens, padding_idx=self.padding_idx)
        outputs.add(nll=(nll_loss, num_words), con_loss=nll_loss, acc=acc, ppl=ppl)
        return outputs
    
    def decode_greedy(self, encoder_states, user_embed, bsz, max_len, data, outputs):
        xs = self._starts(bsz)
        incr_state = None
        logits = []
        for i in range(max_len):
            user_embed_norm = self.user_norm(user_embed)
            latent, incr_state = self.decoder(xs, encoder_states, user_embed_norm, incr_state)
            latent = latent[:, -1:, :]
            # if latent.size(-1) != self.word_embedings_dim:
            #     latent = self.output_layer(latent)
            # latent = self.output_layer(torch.cat([latent, user_embed_norm.unsqueeze(1)], -1))
            pred_logits = F.linear(latent, self.word_embedings.weight)
            _, preds = pred_logits.max(dim=-1)
            logits.append(pred_logits)
            xs = torch.cat([xs, preds], dim=1)
            all_finished = ((xs == self.eos).sum(dim=1) > 0).sum().item() == bsz
            if all_finished:
                break
        logits = torch.cat(logits, 1)
        # con_loss = torch.mean(self.gen_cross_entropy_loss(logits, data.response))
        # outputs.add(con_loss=con_loss)
        return outputs, logits, xs      
    
    def forward(self, data, is_training_con=False, is_generator=False):
        outputs = Pack()
        entity_features = self.RGCN(None, self.edge_idx, self.edge_type)
        
        graph_mask = torch.sign(data.entities_history_vector)
        entities_history_feature = entity_features[data.entities_history_vector]
        entities_history_hidden, star_node_hidden = self.SGNN(entities_history_feature, data.adjacent_matrix, graph_mask)
        get = lambda i: entities_history_hidden[i][data.alias_data[i]]  # look up
        seq_entities_history_hidden = torch.stack([get(i) for i in torch.arange(len(data.alias_data)).long()]) # shape of seq_entities_history_hidden: [batch_size, entity_num, embedding_dim]

        # MIM between star_node and data.total_target (contrastive learning)
        if self.use_mim:
            
            total_target_feature = entity_features[data.total_target]
            total_target_mask = torch.sign(data.total_target).float()
            negative_target_feature = entity_features[data.negative_target]
            pos_score = self.informax_score(star_node_hidden, total_target_feature)
            neg_score = self.informax_score(star_node_hidden, negative_target_feature)
            mim_distance = torch.sigmoid(pos_score - neg_score)
            mim_loss = self.BCELoss(mim_distance, torch.ones_like(mim_distance, dtype=torch.float32))
            mim_loss = torch.sum(mim_loss * total_target_mask.flatten())
            outputs.add(mim_loss=mim_loss)
        
        sequence_emb = self.add_position_embedding(data.total_entities_history, seq_entities_history_hidden)
        masked_entities_sequence = torch.cat([data.total_entities_history[:,:-1], self._target(data.total_entities_history.shape[0])], 1) # add target at last position, [:,:-1] means remove padding at last position
        sequence_star_emb = torch.cat((sequence_emb[:,:-1,:], star_node_hidden.unsqueeze(1)), dim=1) # add star hidden at last position,  [:,:-1,:] means remove padding at last position
        mutiattn_mask = torch.sign(masked_entities_sequence)
        sequence_output = self.entity_encoder(sequence_star_emb, ~mutiattn_mask.bool())
        user_emb = sequence_output[:,0,:]
        # rec
        rating_pred = F.linear(user_emb, entity_features, self.output_en.bias)
        rec_loss = self.CELoss(rating_pred.unsqueeze(1).repeat(1, data.total_target.size(-1),1).view(-1, rating_pred.size(-1)),
                               data.total_target.view(-1))
        rec_mask = torch.sign(data.total_target.view(-1))
        rec_loss = torch.sum(rec_loss * rec_mask.float()) / torch.sum(rec_mask.float() + 1e-30)
        outputs.add(rec_loss=rec_loss)
        
        if not is_training_con:
            return outputs, rating_pred

        uttr_out, uttr_mask = self.encoder(data, entities_history_hidden, star_node_hidden)
        # train conversation
        if is_training_con and not is_generator:
            outputs, logits, preds = self.decode_forced(data, (uttr_out, uttr_mask), user_emb, outputs)
            return outputs, preds
        
        # generator
        if is_training_con and is_generator:
            bsz = uttr_out.size(0)
            outputs, logits, preds = self.decode_greedy((uttr_out, uttr_mask), user_emb, bsz, self.max_response_length, data, outputs)
            return outputs, preds
    
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def collect_metrics(self, outputs, is_training_rec=False, is_training_con=False):
        metrics = Pack()
        loss = 0.0
        if is_training_rec:
            mim_loss = outputs.mim_loss
            rec_loss = outputs.rec_loss
            loss = loss + mim_loss + rec_loss
            metrics.add(mim_loss=mim_loss, rec_loss=rec_loss, loss=loss)
            return metrics
        if is_training_con:
            pass
    
    def iterate(self, data, optimizer=None, grad_clip=None, is_training_rec=False, is_training_con=False):
        outputs, rating_pred = self.forward(data)
        metrics =self.collect_metrics(outputs, is_training_rec=False, is_training_con=False)
        loss = metrics.loss
        
        if torch.isnan(loss):
            raise ValueError("nan loss encountered")
        
        if is_training_rec:
            assert optimizer is not None
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if is_training_con:
            pass
        
        return metrics, outputs
