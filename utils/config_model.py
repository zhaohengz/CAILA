import torch
import torch.optim as optim

from models.caila import CAILA
from models.plain_clip import PlainClip

def configure_model(args, dataset):
    is_open = False

    if args.model == 'CAILA':
        model = CAILA(dataset, args)
        model_params = []
        prompt_params = []
        trainnable_params = ['norm', 'adapter', 'projection', 'gating_fn', 'logit_scale', 'primitive_fusion'] 
        if args.learnable_prompt:
            trainnable_params.append('token_embedding')
        for name, param in model.named_parameters():
            flag = False
            for x in trainnable_params:
                if x in name:
                    param.requires_grad_(True)
                    model_params.append(param)
                    flag = True
                    break
            if flag:
                pass
            elif 'prompt' in name:
                param.requires_grad_(True)
                prompt_params.append(param)
                print("Prompt {}".format(name))
            else:
                param.requires_grad_(False)
        optim_params = [{'params':model_params}, {'params':prompt_params, 'lr': args.lr}]
        optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.wd)
        model.is_open = is_open
    elif args.model == 'plainclip':
        model = PlainClip(dataset, args)
        model_params = []
        prompt_params = []
        trainnable_params = ['norm', 'adapter', 'projection', 'gating_fn', 'logit_scale', 'primitive_fusion'] 
        if args.learnable_prompt:
            trainnable_params.append('token_embedding')
        for name, param in model.named_parameters():
            flag = False
            for x in trainnable_params:
                if x in name:
                    param.requires_grad_(True)
                    model_params.append(param)
                    flag = True
                    break
            if flag:
                pass
            elif 'prompt' in name:
                param.requires_grad_(True)
                prompt_params.append(param)
                print("Prompt {}".format(name))
            else:
                param.requires_grad_(False)
        optim_params = [{'params':model_params}, {'params':prompt_params, 'lr': args.lr}]
        optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.wd)
        model.is_open = is_open
    optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.wd)

    model.is_open = is_open

    return model, optimizer
