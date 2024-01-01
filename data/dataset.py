#external libs
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
import random
from os.path import join as ospj
from glob import glob 
#torch libs
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
#local libs
from utils.utils import get_norm_values, chunks
from itertools import chain, product

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ImageLoader:
    def __init__(self, root):
        self.root_dir = root

    def __call__(self, img):
        img = Image.open(ospj(self.root_dir,img)).convert('RGB') #We don't want alpha
        return img


def dataset_transform(phase, norm_family ='clip'):
    '''
        Inputs
            phase: String controlling which set of transforms to use
            norm_family: String controlling which normaliztion values to use
        
        Returns
            transform: A list of pytorch transforms
    '''
    mean, std = get_norm_values(norm_family=norm_family)

    if phase == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    elif phase == 'val' or phase == 'test':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif phase == 'all':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError('Invalid transform')

    return transform

def filter_data(all_data, pairs_gt, topk = 5):
    '''
    Helper function to clean data
    '''
    valid_files = []
    with open('/home/ubuntu/workspace/top'+str(topk)+'.txt') as f:
        for line in f:
            valid_files.append(line.strip())

    data, pairs, attr, obj  = [], [], [], []
    for current in all_data:
        if current[0] in valid_files:
            data.append(current)
            pairs.append((current[1],current[2]))
            attr.append(current[1])
            obj.append(current[2])
            
    counter = 0
    for current in pairs_gt:
        if current in pairs:
            counter+=1
    print('Matches ', counter, ' out of ', len(pairs_gt))
    print('Samples ', len(data), ' out of ', len(all_data))
    return data, sorted(list(set(pairs))), sorted(list(set(attr))), sorted(list(set(obj)))

# Dataset class now

class CompositionDataset(Dataset):
    '''
    Inputs
        root: String of base dir of dataset
        phase: String train, val, test
        split: String dataset split
        subset: Boolean if true uses a subset of train at each epoch
        num_negs: Int, numbers of negative pairs per batch
        pair_dropout: Percentage of pairs to leave in current epoch
    '''
    def __init__(
        self,
        root,
        phase,
        dataset=None,
        split = 'compositional-split',
        norm_family = 'imagenet',
        subset = False,
        pair_dropout = 0.0,
        return_images = False,
        train_only = False,
        open_world=False
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.pair_dropout = pair_dropout
        self.norm_family = norm_family
        self.return_images = return_images
        self.feat_dim = 2048
        self.open_world = open_world

        self.dataset = dataset

        self.attrs, self.objs, self.pairs, self.train_pairs, \
            self.val_pairs, self.test_pairs = self.parse_split()
        self.train_data, self.val_data, self.test_data = self.get_split_info()
        self.full_pairs = list(product(self.attrs,self.objs))

        # Clean only was here
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr : idx for idx, attr in enumerate(self.attrs)}
        if self.open_world:
            self.pairs = self.full_pairs

        self.all_pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        self.train_only = train_only
        if train_only and self.phase == 'train':
            print('Using only train pairs')
            self.pair2idx = {pair : idx for idx, pair in enumerate(self.train_pairs)}
        else:
            print('Using all pairs')
            self.pair2idx = {pair : idx for idx, pair in enumerate(self.pairs)}
        
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        elif self.phase == 'test':
            self.data = self.test_data
        elif self.phase == 'all':
            print('Using all data')
            self.data = self.train_data + self.val_data + self.test_data
        else:
            raise ValueError('Invalid training phase')
        
        self.all_data = self.train_data + self.val_data + self.test_data
        print('Dataset loaded')
        print('Train pairs: {}, Validation pairs: {}, Test Pairs: {}'.format(
            len(self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('Train images: {}, Validation images: {}, Test images: {}'.format(
            len(self.train_data), len(self.val_data), len(self.test_data)))

        if subset:
            ind = np.arange(len(self.data))
            ind = ind[::len(ind) // 1000]
            self.data = [self.data[i] for i in ind]


        # Keeping a list of all pairs that occur with each object
        self.obj_affordance = {}
        self.train_obj_affordance = {}
        self.attr_affordance = {}
        self.train_attr_affordance = {}
        for _obj in self.objs:
            candidates = [attr for (_, attr, obj) in self.train_data+self.test_data if obj==_obj]
            self.obj_affordance[_obj] = list(set(candidates))

            candidates = [attr for (_, attr, obj) in self.train_data if obj==_obj]
            self.train_obj_affordance[_obj] = list(set(candidates))

        for _attr in self.attrs:
            candidates = [obj for (_, attr, obj) in self.train_data+self.test_data if attr==_attr]
            self.attr_affordance[_attr] = list(set(candidates))

            candidates = [obj for (_, attr, obj) in self.train_data if attr==_attr]
            self.train_attr_affordance[_attr] = list(set(candidates))

        self.sample_indices = list(range(len(self.data)))
        self.selected_indices = []
        self.sample_pairs = self.train_pairs

        # Load based on what to output
        self.transform = dataset_transform(self.phase, self.norm_family)
        self.loader = ImageLoader(ospj(self.root, 'images'))

        self.num_mixup = 7
        self.p_mixup = 0.5
        self.p_shift = 0

        self.pair_dict, self.attr_dict, self.obj_dict = self.build_data_dict(self.data) 

    def build_data_dict(self, data):
        pair_dict = {}
        attr_dict = {}
        obj_dict = {}
        for current in data:
            image, attr, obj = current
            def insert(map, key, value):
                if key not in map:
                    map[key] = []
                map[key].append(value)
            insert(pair_dict, (attr, obj), image)
            insert(attr_dict, attr, image)
            insert(obj_dict, obj, image)
        
        return pair_dict, attr_dict, obj_dict 

    def parse_split(self):
        '''
        Helper function to read splits of object atrribute pair
        Returns
            all_attrs: List of all attributes
            all_objs: List of all objects
            all_pairs: List of all combination of attrs and objs
            tr_pairs: List of train pairs of attrs and objs
            vl_pairs: List of validation pairs of attrs and objs
            ts_pairs: List of test pairs of attrs and objs
        '''
        def parse_pairs(pair_list):
            '''
            Helper function to parse each phase to object attrribute vectors
            Inputs
                pair_list: path to textfile
            '''
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                if self.dataset == 'vaw-czsl':
                    pairs = [t.split('+') for t in pairs]
                else:
                    pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))

            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            ospj(self.root, self.split, 'train_pairs.txt')
        )
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            ospj(self.root, self.split, 'val_pairs.txt')
        )
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            ospj(self.root, self.split, 'test_pairs.txt')
        )
        
        #now we compose all objs, attrs and pairs
        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def get_split_info(self):
        '''
        Helper method to read image, attrs, objs samples

        Returns
            train_data, val_data, test_data: List of tuple of image, attrs, obj
        '''
        data = torch.load(ospj(self.root, 'metadata_{}.t7'.format(self.split)))

        train_data, val_data, test_data = [], [], []

        for instance in data:
            image, attr, obj, settype = instance['image'], instance['attr'], \
                instance['obj'], instance['set']
            curr_data = [image, attr, obj]

            if attr == 'NA' or (attr, obj) not in self.pairs or settype == 'NA':
                # Skip incomplete pairs, unknown pairs and unknown set
                continue

            if settype == 'train':
                train_data.append(curr_data)
            elif settype == 'val':
                val_data.append(curr_data)
            else:
                test_data.append(curr_data)

        return train_data, val_data, test_data

    def get_dict_data(self, data, attrs, objs, pairs):
        data_dict = {}
        for current in objs:
            data_dict[current] = []

        for current in data:
            image, attr, obj = current
            data_dict[obj].append(image)
            # data_dict[(attr, obj)].append(image)
        
        return data_dict


    def reset_dropout(self):
        ''' 
        Helper function to sample new subset of data containing a subset of pairs of objs and attrs
        '''
        self.sample_indices = list(range(len(self.data)))
        self.sample_pairs = self.train_pairs

        # Using sampling from random instead of 2 step numpy
        n_pairs = int((1 - self.pair_dropout) * len(self.train_pairs))

        self.sample_pairs = random.sample(self.train_pairs, n_pairs)
        print('Sampled new subset')
        print('Using {} pairs out of {} pairs right now'.format(
            n_pairs, len(self.train_pairs)))

        self.sample_indices = [ i for i in range(len(self.data))
            if (self.data[i][1], self.data[i][2]) in self.sample_pairs
        ]
        print('Using {} images out of {} images right now'.format(
            len(self.sample_indices), len(self.data)))

    def sample_negative(self, attr, obj):
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Returns
            Tuple of a different attribute, object indexes
        '''
        new_attr, new_obj = self.sample_pairs[np.random.choice(
            len(self.sample_pairs))]

        while new_attr == attr and new_obj == obj:
            new_attr, new_obj = self.sample_pairs[np.random.choice(
            len(self.sample_pairs))]
        
        return (self.attr2idx[new_attr], self.obj2idx[new_obj])
    
    def sample_mixup(self, attr, obj):
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Returns
            Tuple of a different attribute, object indexes
        '''
        idx = np.random.choice(len(self.sample_indices))
        _, new_attr, new_obj = self.data[self.sample_indices[idx]]

        while (new_attr == attr and new_obj == obj):
            idx = np.random.choice(len(self.sample_indices))
            _, new_attr, new_obj = self.data[self.sample_indices[idx]]
        
        return self.attr2idx[new_attr], self.obj2idx[new_obj], self.pair2idx[(new_attr, new_obj)], idx

    def sample_affordance(self, attr, obj):
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Return
            Idx of a different attribute for the same object
        '''
        new_attr = np.random.choice(self.obj_affordance[obj])
        
        while new_attr == attr:
            new_attr = np.random.choice(self.obj_affordance[obj])
        
        return self.attr2idx[new_attr], self.pair2idx[(new_attr, obj)]

    def sample_train_affordance(self, attr, obj, map, target):
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Return
            Idx of a different attribute for the same object from the training pairs
        '''
        if target == 'attr':
            new_attr = np.random.choice(self.train_obj_affordance[obj])
            
            while new_attr == attr and len(self.train_obj_affordance[obj]) > 1:
                new_attr = np.random.choice(self.train_obj_affordance[obj])

            sample = random.sample(map[new_attr], 1)

            return self.attr2idx[new_attr], self.pair2idx[(new_attr, obj)], sample
        
        elif target == 'obj':
            new_obj = np.random.choice(self.train_attr_affordance[attr])
            
            while new_obj == obj and len(self.train_attr_affordance[attr]) > 1:
                new_obj = np.random.choice(self.train_attr_affordance[attr])

            sample = random.sample(map[new_obj], 1)

            return self.obj2idx[new_obj], self.pair2idx[(attr, new_obj)], sample

    def set_p(self, p_mixup, p_shift, p_obj_shift):
        self.p_mixup = p_mixup
        self.p_shift = p_shift
        self.p_obj_shift = p_obj_shift
    
    def sample_contrastive(self, map, key, num_neg):

        postive = map.pop(key)
        pos_sample = random.choice(postive)
        negative = list(chain(*map.values()))
        neg_sample = random.sample(negative, num_neg)

        map[key] = postive

        samples = [pos_sample] + neg_sample

        return samples


    def __getitem__(self, index):
        '''
        Call for getting samples
        '''
        index = self.sample_indices[index]

        image, attr, obj = self.data[index]

        img = self.loader(image)
        img = self.transform(img)

        data = [img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]]
        
        if self.phase == 'train':

            # Sample mixup
            p_shift = random.random()
            do_shift = True if p_shift < self.p_shift else False
            if not do_shift:
                p_mixup = random.random()
                do_mixup = True if p_mixup < self.p_mixup else False
            else:
                do_mixup = False

            mixup_prob = torch.zeros(self.num_mixup + 1)
            if do_mixup: 
                mixup_prob[0] = random.random()
                all_mixup_attrs = [self.attr2idx[attr]]
                all_mixup_objs = [self.obj2idx[obj]]
                all_mixup_pairs = [self.pair2idx[(attr, obj)]]
                idx_to_mix = []
                for i in range(self.num_mixup):
                    mixup_prob[i+1] = random.random()
                    mix_attr, mix_obj, mix_pair, idx = self.sample_mixup(attr, obj)
                    all_mixup_attrs.append(mix_attr)
                    all_mixup_objs.append(mix_obj)
                    all_mixup_pairs.append(mix_pair)
                    idx_to_mix.append(idx)

                all_mixup_attrs = torch.as_tensor(all_mixup_attrs)
                all_mixup_objs = torch.as_tensor(all_mixup_objs)
                all_mixup_pairs = torch.as_tensor(all_mixup_pairs)
            
                mixup_prob = mixup_prob / mixup_prob.sum()

                mixup_img = [img]
                for idx in idx_to_mix:
                    data_idx = self.sample_indices[idx]
                    image, attr, obj = self.data[data_idx]
                    img = self.loader(image)
                    img = self.transform(img)
                    mixup_img.append(img)
                
                mixup_img = (torch.stack(mixup_img, dim=0) * mixup_prob.view(-1,1,1,1)).sum(dim=0)
            else:
                mixup_prob = torch.zeros(self.num_mixup + 1)
                mixup_prob[0] = 1
                all_mixup_attrs = torch.zeros(self.num_mixup + 1).long()
                all_mixup_objs = torch.zeros(self.num_mixup + 1).long()
                all_mixup_pairs = torch.zeros(self.num_mixup + 1).long()
                mixup_img = img

            data  += [all_mixup_attrs, all_mixup_objs, all_mixup_pairs, do_mixup, mixup_prob]
            data[0] = mixup_img

            if do_shift:
                if self.train_only:
                    p_obj = random.random()
                    do_attr_shift = False
                    do_obj_shift = False
                    if p_obj < self.p_obj_shift:
                        new_obj, new_pair, sample = self.sample_train_affordance(attr, obj, self.obj_dict, target='obj')
                        do_obj_shift = True
                        new_attr = self.attr2idx[attr]
                    else:
                        new_attr, new_pair, sample = self.sample_train_affordance(attr, obj, self.attr_dict, target='attr')
                        do_attr_shift = True
                        new_obj = self.obj2idx[obj]

                    if sample:
                        sample = sample[0]
                        sample = self.loader(sample)
                        sample = self.transform(sample)
                    else:
                        sample = data[0]
                    data += [new_obj, new_attr, new_pair, sample, do_obj_shift, do_attr_shift]
            else:
                data += [self.obj2idx[obj], self.attr2idx[attr], self.pair2idx[(attr, obj)], data[0], 0, 0]

        # Return image paths if requested as the last element of the list
        if self.return_images and self.phase != 'train':
            data.append(image)

        return data
    
    def __len__(self):
        '''
        Call for length
        '''
        return len(self.sample_indices)
