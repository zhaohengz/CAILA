import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from models.losses import MixupClassLoss
from models.clip import CLIPModel
from transformers.models.clip.configuration_clip import CLIPConfig
from transformers import CLIPProcessor

def adjust_weights(adj, embeddings, offset):
    for idx in range(offset, adj.shape[0]):
        valid_edges = adj[idx].nonzero()[0]
        for v in valid_edges:
            adj[idx, v] = 0
            adj[v, idx] = 0
    for idx in range(adj.shape[0]):
        adj[idx, idx] = 30
    return adj

def clean_text(v):
    custom_map = {
        'Faux.Fur': 'fake fur',
        'Faux.Leather': 'fake leather',
        'Full.grain.leather': 'thick leather',
        'Hair.Calf': 'hairy leather',
        'Patent.Leather': 'shiny leather',
        'Boots.Ankle': 'ankle boots',
        'Boots.Knee.High': 'kneehigh boots',
        'Boots.Mid-Calf': 'midcalf boots',
        'Shoes.Boat.Shoes': 'boatshoes',
        'Shoes.Clogs.and.Mules': 'clogs shoes',
        'Shoes.Flats': 'flats shoes',
        'Shoes.Heels': 'heels',
        'Shoes.Loafers': 'loafers',
        'Shoes.Oxfords': 'oxford shoes',
        'Shoes.Sneakers.and.Athletic.Shoes': 'sneakers',
        'traffic_light': 'traficlight',
        'trash_can': 'trashcan',
        'dry-erase_board' : 'dry_erase_board',
        'black_and_white' : 'black_white',
        'eiffel_tower' : 'tower'
    }

    if v in custom_map:
        return custom_map[v]
    else:
        return v.lower()

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class CAILA(nn.Module):
    def __init__(self, dset, args):
        super(CAILA, self).__init__()
        self.args = args
        self.dset = dset
        
        # Image Embedder
        self.num_attrs, self.num_objs, self.num_pairs = len(dset.attrs), len(dset.objs), len(dset.pairs)

        self.all_attrs = []
        self.all_objs = []

        for pair in self.dset.pairs:
            attr, obj = pair
            self.all_attrs.append(self.dset.attr2idx[attr])
            self.all_objs.append(self.dset.obj2idx[obj])
        
        unseen_in_vocab_idx = []
        for pair in self.dset.val_pairs + self.dset.test_pairs:
            if pair in self.dset.train_pairs:
                pass
            else:
                unseen_in_vocab_idx.append(self.dset.all_pair2idx[pair])
        self.unseen_in_vocab_idx = torch.LongTensor(unseen_in_vocab_idx)

        if self.args.train_only:
            train_idx = []
            self.all_train_attrs = []
            self.all_train_objs = []
            self.train_relations = torch.zeros((len(dset.train_pairs), len(dset.train_pairs)))
            for current in dset.train_pairs:
                train_idx.append(dset.all_pair2idx[current])
                attr, obj = current
                self.all_train_attrs.append(self.dset.attr2idx[attr])
                self.all_train_objs.append(self.dset.obj2idx[obj])
            self.train_idx = torch.LongTensor(train_idx)
            for i, p0 in enumerate(dset.train_pairs):
                for j, p1 in enumerate(dset.train_pairs):
                    if (i == j):
                        continue
                    attr0, obj0 = p0
                    attr1, obj1 = p1
                    if (attr0 == attr1) or (obj0 == obj1):
                        self.train_relations[i][j] = 1
            
            oov_idx = list(set([x for x in range(self.num_pairs)]) - set(train_idx) - set(unseen_in_vocab_idx))
            self.oov_idx = torch.LongTensor(oov_idx)
        else:
            self.all_train_attrs = self.all_attrs
            self.all_train_objs = self.all_objs
            self.train_idx = torch.LongTensor(list(dset.all_pair2idx.values()))

        self.open_world = args.open_world
        self.fusion_start_layer = args.fusion_start_layer
        
        config = CLIPConfig.from_pretrained("openai/{}".format(args.clip_config))
        config.text_config.reduction_factor = args.reduction_factor
        config.vision_config.reduction_factor = args.reduction_factor 
        
        config.text_config.track_z = False
        config.text_config.adapter_modes= ['pair', 'obj','attr']
        config.text_config.has_adapter = args.enable_text_adapter
        config.text_config.mixture_of_adapters = False

        config.vision_config.adapter_modes= ['obj', 'attr']
        config.vision_config.track_z = False
        config.vision_config.mixture_of_adapters = True
        config.vision_config.has_adapter = args.enable_vision_adapter
        config.vision_config.combine_latent = args.combine_latent
        config.vision_config.combine_output = args.combine_output
        config.vision_config.combination_ops = args.combination_ops
        config.vision_config.fusion_start_layer = args.fusion_start_layer
        config.vision_config.adapter_start_layer = args.adapter_start_layer
        config.vision_config.adapter_end_layer = args.adapter_end_layer
        config.vision_config.fusion_key = 'mixture'

        self.clip_model = CLIPModel(config)
        try:
            checkpoint = torch.load('./clip_ckpts/{}.pth'.format(args.clip_config), map_location='cpu')
            msg = self.clip_model.load_state_dict(checkpoint, strict=False)
        except:
            print('Could not load local clip weights, initializing from scratch')

        self.processor = CLIPProcessor.from_pretrained("openai/{}".format(args.clip_config))

        pairs = [' '.join([clean_text(t) for t in c]) for c in self.dset.pairs]
        self.pair_inputs = self.processor(text=[f"a photo of a {c}" for c in pairs], return_tensors="pt", padding=True)
        self.attr_inputs = self.processor(text=["a photo of a {} object".format(clean_text(c)) for c in self.dset.attrs], return_tensors="pt", padding=True)
        self.obj_inputs = self.processor(text=["a photo of a {}".format(clean_text(c)) for c in self.dset.objs], return_tensors="pt", padding=True)

        self.prompt_loc = 5

        self.dropout = nn.Dropout(args.img_dropout)
        
        self.attr_logit_scale = nn.Parameter(torch.ones([]) * (self.clip_model.logit_scale + math.log(20.0)))
        self.obj_logit_scale = nn.Parameter(torch.ones([]) * (self.clip_model.logit_scale + math.log(20.0)))

        self.mixup_loss = MixupClassLoss()

        self.saved_pair_embeds = None

    def reset_saved_pair_embeds(self):
        self.saved_pair_embeds = None

    def forward(self, x):
        if self.training:
            loss, pred = self.run(x)
            return loss, pred
        else:
            with torch.no_grad():
                scores = self.run(x)
            return None, scores
    
    def apply_gating_fn(self, pair_embeds, attr_embeds, obj_embeds):
        return torch.stack([pair_embeds, attr_embeds, obj_embeds], dim=-1).mean(dim=-1)

        
    def tensor_projection(self, src, dst):
        # make projection from src to dst
        
        dst_norm = F.normalize(dst, dim=-1, p=2)
        projection = torch.bmm(src.unsqueeze(1), dst.unsqueeze(-1)).squeeze(-1) * dst_norm

        return projection

    def run(self, x):
        img = x[0]

        device = img.device

        pos_attr_feats, attr_vision_outputs = self.clip_model.get_image_features(
            img, 
            mode='attr', 
            output_hidden_states=True,
            output_pre_adapter_hidden_states=True,
        )
        pos_obj_feats, obj_vision_outputs = self.clip_model.get_image_features(
            img, 
            mode='obj',
            output_hidden_states=True,
            output_pre_adapter_hidden_states=True,
        )

        attr_adapter_feats, attr_hidden_states = attr_vision_outputs.pre_adapter_hidden_states, attr_vision_outputs.hidden_states
        obj_adapter_feats, obj_hidden_states = obj_vision_outputs.pre_adapter_hidden_states, obj_vision_outputs.hidden_states

        attr_embeds = self.clip_model.get_text_features(
            **self.attr_inputs.to(device), 
            prompt_loc=self.prompt_loc,
            mode='attr')

        obj_embeds = self.clip_model.get_text_features(
            **self.obj_inputs.to(device), 
            prompt_loc=self.prompt_loc,
            mode='obj'
        )

        if self.training:
            attrs, objs, pairs = x[1], x[2], x[3]
            mixup_attrs, mixup_objs, mixup_pairs, do_mixup, mixup_prob = x[4], x[5], x[6], x[7], x[8]
            new_objs, new_attrs, new_pairs, new_sample, do_obj_shift, do_attr_shift = x[9], x[10], x[11], x[12], x[13], x[14]

            # Perform attr shift 
            indices = do_attr_shift.nonzero().squeeze(-1)
            new_attr_sample = new_sample[indices]

            if len(indices) > 0:
                shifted_attr_adapter_feats = [[x[0].clone(), x[1].clone()] for x in attr_adapter_feats]
                shifted_attr_hidden_states = [x.clone() for x in attr_hidden_states]
                target_attr_feats, target_attr_vision_outputs = self.clip_model.get_image_features(
                    new_attr_sample,
                    mode='attr',
                    output_pre_adapter_hidden_states=True,
                    output_hidden_states=True
                )
                target_attr_adapter_feats, target_attr_hidden_states = target_attr_vision_outputs.pre_adapter_hidden_states, target_attr_vision_outputs.hidden_states
                for layer, feat in enumerate(target_attr_adapter_feats):
                    shifted_attr_adapter_feats[layer][0][indices] = feat[0]
                    shifted_attr_adapter_feats[layer][1][indices] = feat[1]
                    shifted_attr_hidden_states[layer][indices] = target_attr_hidden_states[layer]
                
                pos_attr_feats[indices] = target_attr_feats
                attrs[indices] = new_attrs[indices]
                pairs[indices] = new_pairs[indices]
            else:
                shifted_attr_adapter_feats = attr_adapter_feats
                shifted_attr_hidden_states = attr_hidden_states

            # Perform obj shift
            indices = do_obj_shift.nonzero().squeeze(-1)
            new_obj_sample = new_sample[indices]

            if len(indices) > 0:
                shifted_obj_adapter_feats = [[x[0].clone(), x[1].clone()] for x in obj_adapter_feats]
                shifted_obj_hidden_states = [x.clone() for x in obj_hidden_states]
                target_obj_feats, target_obj_vision_outputs = self.clip_model.get_image_features(
                    new_obj_sample,
                    mode='obj',
                    output_pre_adapter_hidden_states=True,
                    output_hidden_states=True
                )
                target_obj_adapter_feats, target_obj_hidden_states = target_obj_vision_outputs.pre_adapter_hidden_states, target_obj_vision_outputs.hidden_states
                for layer, feat in enumerate(target_obj_adapter_feats):
                    shifted_obj_adapter_feats[layer][0][indices] = feat[0]
                    shifted_obj_adapter_feats[layer][1][indices] = feat[1]
                    shifted_obj_hidden_states[layer][indices] = target_obj_hidden_states[layer]
                
                pos_obj_feats[indices] = target_obj_feats
                objs[indices] = new_objs[indices]
                pairs[indices] = new_pairs[indices]
            else:
                shifted_obj_adapter_feats = obj_adapter_feats
                shifted_obj_hidden_states = obj_hidden_states

            do_mixup = do_mixup.long()
      
            train_pair_inputs = {k: v[self.train_idx.cpu()].to(device) for k,v in self.pair_inputs.items()}

            pair_embeds = self.clip_model.get_text_features(
                **train_pair_inputs,
                prompt_loc=self.prompt_loc,
                mode='pair'
            )

            num_train_pairs = len(self.all_train_attrs)           
            pairwise_attr_embeds = attr_embeds[self.all_train_attrs]
            pairwise_obj_embeds = obj_embeds[self.all_train_objs]
        else:
            shifted_attr_feats = None
            if self.saved_pair_embeds is not None:
                pair_embeds = self.saved_pair_embeds
            else:
                num_pairs = len(self.all_attrs)
                if num_pairs > 25000:
                    embeds = []
                    for start in range(0, num_pairs, 25000):
                        end = min(start + 25000, num_pairs)
                        pair_inputs = {k: v[start:end].to(device) for k,v in self.pair_inputs.items()}
                        pair_embeds = self.clip_model.get_text_features(
                            **pair_inputs,
                            mode='pair'
                        )
                        embeds.append(pair_embeds) 
                    pair_embeds = torch.cat(embeds, dim=0)
                else:
                    pair_embeds = self.clip_model.get_text_features(
                        **self.pair_inputs.to(device),
                        mode='pair'
                    )
                self.saved_pair_embeds = pair_embeds
            pairwise_attr_embeds = attr_embeds[self.all_attrs]
            pairwise_obj_embeds = obj_embeds[self.all_objs]

        if self.training:
            img_feats = self.clip_model.get_image_features(
                inputs_embeds=torch.cat([shifted_attr_hidden_states[self.fusion_start_layer], shifted_obj_hidden_states[self.fusion_start_layer]], dim=1),
                mode='mixture',
                concept_hidden_states=[[a, o] for a, o in zip(shifted_attr_adapter_feats, shifted_obj_adapter_feats)],
                another_cls=attr_hidden_states[self.fusion_start_layer].shape[1]
            )[0]
        else:
            img_feats = self.clip_model.get_image_features(
                inputs_embeds=torch.cat([attr_hidden_states[self.fusion_start_layer], obj_hidden_states[self.fusion_start_layer]], dim=1),
                mode='mixture',
                concept_hidden_states=[[a, o] for a, o in zip(attr_adapter_feats, obj_adapter_feats)],
                another_cls=attr_hidden_states[self.fusion_start_layer].shape[1]
            )[0]

        img_feats = F.normalize(img_feats, dim=-1, p=2)
        img_feats = self.dropout(img_feats)
        pos_obj_feats = F.normalize(pos_obj_feats, dim=-1, p=2)
        pos_attr_feats = F.normalize(pos_attr_feats, dim=-1, p=2)

        pair_embeds = self.apply_gating_fn(pair_embeds, pairwise_attr_embeds, pairwise_obj_embeds)
        pair_logit_scale = self.clip_model.logit_scale.exp()
        obj_logit_scale = self.obj_logit_scale.exp()
        attr_logit_scale = self.attr_logit_scale.exp()

        pair_embeds = F.normalize(pair_embeds, dim=-1, p=2)


        if self.training:
            pair_embeds = pair_embeds.permute(1, 0)
            pair_pred = torch.matmul(img_feats, pair_embeds) * pair_logit_scale

            obj_embeds = F.normalize(obj_embeds, dim=-1, p=2).permute(1, 0)
            attr_embeds = F.normalize(attr_embeds, dim=-1, p=2).permute(1, 0)

            obj_pred = torch.matmul(pos_obj_feats, obj_embeds) * obj_logit_scale
            attr_pred = torch.matmul(pos_attr_feats, attr_embeds) * attr_logit_scale

            loss = self.mixup_loss(pair_pred, pairs, mixup_pairs, do_mixup, mixup_prob) + \
                1 * self.mixup_loss(obj_pred, objs, mixup_objs, do_mixup, mixup_prob) + \
                1 * self.mixup_loss(attr_pred, attrs, mixup_attrs, do_mixup, mixup_prob) 

            return loss, None
        else:
            pair_embeds = pair_embeds.permute(1, 0)
            score = torch.matmul(img_feats, pair_embeds) 
            return score.cpu()
       
