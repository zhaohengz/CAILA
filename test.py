#  Torch imports
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import numpy as np
from flags import DATA_FOLDER

cudnn.benchmark = True

# Python imports
import tqdm
from tqdm import tqdm
import os
from os.path import join as ospj

# Local imports
from data import dataset as dset
from models.common import Evaluator
from utils.utils import load_args
from utils.config_model import configure_model
from flags import parser



device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # Get arguments and start logging
    args = parser.parse_args()
    logpath = args.logpath
    load_args(args.config, args)

    # Get dataset
    trainset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER, args.data_dir),
        phase='train',
        split=args.splitname,
        train_only=args.train_only,
        subset=args.subset,
        open_world=args.open_world,
        dataset=args.dataset
    )

    testset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='test',
        split=args.splitname,
        subset=args.subset,
        open_world=args.open_world,
        norm_family=args.norm_family,
        dataset=args.dataset
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers)


    # Get model and optimizer
    model, _ = configure_model(args, trainset)


    args.load = ospj(logpath,'ckpt_best_auc.t7')

    checkpoint = torch.load(args.load)
    model.load_state_dict(checkpoint['net'], strict=True)
    model = model.cuda()
    model.eval()

    evaluator = Evaluator(testset, model)

    with torch.no_grad():
        test(model, testloader, evaluator, args)


def test(model, testloader, evaluator,  args, threshold=None, print_results=True):

        model.eval()
        accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []

        for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
            data = [d.to(device) for d in data]

            _, predictions = model(data)

            attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]

            all_pred.append(predictions)
            all_attr_gt.append(attr_truth)
            all_obj_gt.append(obj_truth)
            all_pair_gt.append(pair_truth)

        if args.cpu_eval:
            all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt), torch.cat(all_obj_gt), torch.cat(all_pair_gt)
        else:
            all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
                'cpu'), torch.cat(all_pair_gt).to('cpu')

        # Calculate best unseen accuracy
        all_pred_dict = torch.cat(all_pred, dim=0)
        results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=args.bias, topk=args.topk)
        stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict,
                                               topk=args.topk)

        result = ''
        for key in stats:
            result = result + key + '  ' + str(round(stats[key], 4)) + '| '

        result = result + args.name
        if print_results:
            print(f'Results')
            print(result)
        return results


if __name__ == '__main__':
    main()
