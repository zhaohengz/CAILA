import argparse

DATA_FOLDER = "./all_data"
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--config', default='configs/args.yml', help='path of the config file (training only)')
parser.add_argument('--dataset', default='mitstates', help='mitstates|zappos')
parser.add_argument('--data_dir', default='mit-states', help='local path to data root dir from ' + DATA_FOLDER)
parser.add_argument('--logpath', default=None, help='Path to dir where to logs are stored (test only)')
parser.add_argument('--splitname', default='compositional-split-natural', help="dataset split")
parser.add_argument('--cv_dir', default='logs/', help='dir to save checkpoints and logs to')
parser.add_argument('--name', default='temp', help='Name of exp used to name models')
parser.add_argument('--load', default=None, help='path to checkpoint to load from')
parser.add_argument('--norm_family', default = 'imagenet', help = 'Normalization values from dataset')
parser.add_argument('--num_negs', type=int, default=1, help='Number of negatives to sample per positive (triplet loss)')
parser.add_argument('--pair_dropout', type=float, default=0.0, help='Each epoch drop this fraction of train pairs')
parser.add_argument('--test_set', default='val', help='val|test mode')
parser.add_argument('--clean_only', action='store_true', default=False, help='use only clean subset of data (mitstates)')
parser.add_argument('--subset', action='store_true', default=False, help='test on a 1000 image subset (debug purpose)')
parser.add_argument('--open_world', action='store_true', default=False, help='perform open world experiment')
parser.add_argument('--test_batch_size', type=int, default=32, help="Batch size at test/eval time")
parser.add_argument('--cpu_eval', action='store_true', help='Perform test on cpu')

# Model parameters
parser.add_argument('--model', default='graphfull', help='visprodNN|redwine|labelembed+|attributeop|tmn|compcos')
parser.add_argument('--bias', type=float, default=1e3, help='Bias value for unseen concepts')
parser.add_argument('--train_only', action='store_true', default=False, help='Optimize only for train pairs')

# Hyperparameters
parser.add_argument('--topk', type=int, default=1,help="Compute topk accuracy")
parser.add_argument('--workers', type=int, default=8,help="Number of workers")
parser.add_argument('--batch_size', type=int, default=512,help="Training batch size")
parser.add_argument('--lr', type=float, default=5e-5,help="Learning rate")
parser.add_argument('--wd', type=float, default=5e-5,help="Weight decay")
parser.add_argument('--save_every', type=int, default=10000,help="Frequency of snapshots in epochs")
parser.add_argument('--eval_val_every', type=int, default=1,help="Frequency of eval in epochs")
parser.add_argument('--max_epochs', type=int, default=800,help="Max number of epochs")

parser.add_argument("--mixup_ratio", default=0)
parser.add_argument("--img_dropout", default=0.0)
parser.add_argument("--reduction_factor", default=4, type=int)
parser.add_argument("--clip_config", default='clip-vit-base-patch32')

parser.add_argument('--device', default='cuda')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--distributed', default=True, type=bool)

parser.add_argument("--concept_shift_prob", default=0)
parser.add_argument("--obj_shift_ratio", default=0.5)
parser.add_argument("--lambda_hsic", default=0, type=float)
parser.add_argument("--learnable_prompt", default=True, type=bool)
parser.add_argument("--fusion_start_layer", default=6, type=int)
parser.add_argument("--fusion_end_layer", default=100, type=int)
parser.add_argument("--adapter_start_layer", default=0, type=int)
parser.add_argument("--adapter_end_layer", default=100, type=int)
parser.add_argument("--combine_latent", default=True, type=bool)
parser.add_argument("--combine_output", default=True, type=bool)
parser.add_argument("--combination_ops", default='mean', type=str)

parser.add_argument("--enable_text_adapter", default=True, type=bool)
parser.add_argument("--enable_vision_adapter", default=True, type=bool)

parser.add_argument('--fp16', default=True, type=bool)
