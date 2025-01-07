import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from trainer_unet import trainer_synapse
import copy


from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/Synapse', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--output_dir', type=str, help='output dir')                   
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=128, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=2e-3,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
# parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--gpu_id',default=0,type=int)
parser.add_argument('--lambda_x',default=0.015,type=float)
parser.add_argument('--dino_weight',default=0.3,type=float)
parser.add_argument('--alpha',default=20.,type=float)
parser.add_argument('--sigma',default=5.,type=float)

# lambda_x

args = parser.parse_args()
if args.dataset == "Synapse":
    args.root_path = os.path.join(args.root_path, "train_npz")
    
def load_from(model, ckpt_path):
    pretrained_path = ckpt_path
    if pretrained_path is not None:
        print("pretrained_path:{}".format(pretrained_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(pretrained_path, map_location=device)

        pretrained_dict = pretrained_dict['model']
        print("---start load pretrained modle of swin encoder---")
        # print(pretrained_dict.keys())
        model_dict = model.state_dict()
        # print(model_dict.keys())
        
        full_dict = copy.deepcopy(pretrained_dict)
        for k, v in pretrained_dict.items():
            if "network." in k:
                up_layer_num = 6-int(k.split('.')[1])
                current_k_up = "network_up_layers." + str(up_layer_num) + '.' + '.'.join(k.split('.')[2:])
                full_dict.update({current_k_up:v})
                full_dict["network_down_layers." + '.'.join(k.split('.')[1:])] = full_dict.pop(k)

        for k in list(full_dict.keys()):
            if k in model_dict:
                if full_dict[k].shape != model_dict[k].shape:
                    print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                    del full_dict[k]


        msg = model.load_state_dict(full_dict, strict=False)
        print(msg)
        return model
    else:
        print("none pretrain")


if __name__ == "__main__":
    torch.cuda.set_device(args.gpu_id)
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    from unet import Eff_Unet
    net = Eff_Unet(
        layers=[5, 5, 15, 10],
        embed_dims=[40, 80, 192, 384],
        downsamples=[True, True, True, True],
        vit_num=6,
        drop_path_rate=0.1,
        num_classes=9).cuda()
    
    net = load_from(net, 'eformer_l_450.pth')


    trainer = {'Synapse': trainer_synapse,}
    trainer[dataset_name](args, net, args.output_dir)