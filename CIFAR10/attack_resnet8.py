import argparse
import copy
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp


from fast_adversarial.CIFAR10.utils import (upper_limit, lower_limit, std, clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard)
from neural_networks.CIFAR10.resnet import resnet8, resnet20, resnet32, resnet56
from neural_networks.utils import load_scaling_factors
from neural_networks.adapt.approx_layers.axx_layers import AdaPT_Conv2d
logger = logging.getLogger(__name__)


mult_index = 0
def update_layer(in_module, name, mult_type):
    global mult_index
    new_module = False
    if isinstance(in_module, AdaPT_Conv2d):
        print(f'mult_type = {mult_type}')
        new_module = True
        in_module.axx_mult = mult_type
        in_module.update_kernel()
        mult_index += 1
    return new_module

def update_model(model, mult_base, appr_level_list):
    global mult_index
    for name, module in model.named_children():
        mult_type = mult_base + str(appr_level_list[mult_index])
        new_module = update_layer(module, name, mult_type)
        if not new_module:
            update_model(module, mult_base, appr_level_list)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='./data', type=str)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=10, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous'],
        help='Perturbation initialization method')
    parser.add_argument('--out-dir', default='./fast_adversarial/log', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--early-stop', action='store_true', help='Early stop if overfitting occurs')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    parser.add_argument('--neural-network', default="resnet8", type=str, help="Choose one from resnet8, resnet20, resnet32, resnet56")
    parser.add_argument('--act-bit', default=8, type=int, help="activation precision used for all layers")
    parser.add_argument('--weight-bit', default=8, type=int, help="weight precision used for all layers")
    parser.add_argument('--bias-bit', default=32, type=int, help="bias precision used for all layers")
    parser.add_argument('--fake-quant', default=True, type=bool, help="Set to True to use fake quantization, set to False to use integer quantization")
    parser.add_argument('--activation-function', default="ReLU", type=str, help="Activation function used for each act layer.")
    parser.add_argument('--execution-type', default='quant', type=str, help="Select type of neural network and precision. Options are: float, quant, adapt. \n float: the neural network is executed with floating point precision.\n quant: the neural network weight, bias and activations are quantized to 8 bit\n adapt: the neural network is quantized to 8 bit and processed with exact/approximate multipliers")
    parser.add_argument('--appr-level', default=0, type=int, help="Approximation level used in all layers (0 is exact)")
    parser.add_argument('--appr-level-list', type=int, nargs=8, help="Exactly 8 integers specifying levels of approximation for each layer")
    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    
    if args.appr_level_list is None:
        approximation_levels = [args.appr_level, args.appr_level, args.appr_level, args.appr_level, args.appr_level, args.appr_level, args.appr_level, args.appr_level]
    else:
        approximation_levels = args.appr_level_list

    model_dir = "./neural_networks/models/"

    if args.execution_type == "quant" or args.execution_type == "adapt":
        namebit = "_a"+str(args.act_bit)+"_w"+str(args.weight_bit)+"_b"+str(args.bias_bit)
    else:
        namebit = ""

    if args.fake_quant:
        namequant = "_fake"
    else:
        namequant = "_int"
    dataset="cifar10"
    if args.execution_type == "float":
        execution_type = "float"
    else:
        execution_type = "quant"
    dataset="cifar10"
    num_classes=10
    filename = model_dir + args.neural_network + namebit + namequant + "_" + execution_type + "_" + dataset + "_" + args.activation_function + ".pth"
    filename_sc = model_dir + args.neural_network + namebit + namequant + "_" + execution_type + "_" + dataset +"_" + args.activation_function + '_scaling_factors.pkl'

    output_log = "no_AT_" + args.neural_network + namebit + namequant + "_" + args.execution_type + "_" + dataset + "_" + args.activation_function  + "_alpha" + str(args.alpha) +"_epsilon" + str(args.epsilon) + "_" + str(args.epochs) + ".log"
    logfile = os.path.join(args.out_dir, output_log)
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(args.out_dir, output_log))
    logger.info(args)

    if args.execution_type == 'adapt':
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    
    mode= {"execution_type":args.execution_type, "act_bit":args.act_bit, "weight_bit":args.weight_bit, "bias_bit":args.bias_bit, "fake_quant":args.fake_quant, "classes":num_classes, "act_type":args.activation_function}
    # Evaluation
    if args.neural_network == "resnet8":
        model_test = resnet8(mode).to(device)
    elif args.neural_network == "resnet20":
        model_test = resnet20(mode).to(device)
    elif args.neural_network == "resnet32":
        model_test = resnet32(mode).to(device)
    elif args.neural_network == "resnet56":
        model_test = resnet56(mode).to(device)
    else:
        exit("error unknown CNN model name")
    checkpoint = torch.load(filename, map_location=device)
    model_test.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model_test.float()
    model_test.eval()
    load_scaling_factors(model_test, filename_sc, device)

    base_mult = "bw_mult_9_9_"
    update_model(model_test, base_mult, approximation_levels)
    print(f"Original device = {device}")
    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 10, device)
    test_loss, test_acc = evaluate_standard(test_loader, model_test, device)

    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)


if __name__ == "__main__":
    main()
