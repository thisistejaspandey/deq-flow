from __future__ import division, print_function
import sys
sys.path.append('core')
from deq_flow import DEQFlow
from utils import flow_viz
from utils.utils import InputPadder
from deq.arg_utils import add_deq_args

sys.path.append("/media/raid/gits/nuscenes_api")
from nuscenes_dataset import NuScenesDataset

import torch
from torch import nn
import numpy as np
import cv2

import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help="Enable Eval mode.")
    parser.add_argument('--test', action='store_true', help="Enable Test mode.")
    parser.add_argument('--viz', action='store_true', help="Enable Viz mode.")
    parser.add_argument('--fixed_point_reuse', action='store_true', help="Enable fixed point reuse.")
    parser.add_argument('--warm_start', action='store_true', help="Enable warm start.")

    parser.add_argument('--name', default='deq-flow', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 

    parser.add_argument('--total_run', type=int, default=1, help="total number of runs")
    parser.add_argument('--start_run', type=int, default=1, help="begin from the given number of runs")
    parser.add_argument('--restore_name', help="restore experiment name")
    parser.add_argument('--resume_iter', type=int, default=-1, help="resume from the given iterations")

    parser.add_argument('--tiny', action='store_true', help='use a tiny model for ablation study')
    parser.add_argument('--large', action='store_true', help='use a large model')
    parser.add_argument('--huge', action='store_true', help='use a huge model')
    parser.add_argument('--gigantic', action='store_true', help='use a gigantic model')
    parser.add_argument('--old_version', action='store_true', help='use the old design for flow head')

    parser.add_argument('--restore_ckpt', help="restore checkpoint for val/test/viz")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--test_set', type=str, nargs='+')
    parser.add_argument('--viz_set', type=str, nargs='+')
    parser.add_argument('--viz_split', type=str, nargs='+', default=['test'])
    parser.add_argument('--output_path', help="output path for evaluation")

    parser.add_argument('--eval_interval', type=int, default=5000, help="evaluation interval")
    parser.add_argument('--save_interval', type=int, default=5000, help="saving interval")
    parser.add_argument('--time_interval', type=int, default=500, help="timing interval")

    parser.add_argument('--gma', action='store_true', help='use gma')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--schedule', type=str, default="onecycle", help="learning rate schedule")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--vdropout', type=float, default=0.0, help="variational dropout added to BasicMotionEncoder for DEQs")
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--active_bn', action='store_true')
    parser.add_argument('--all_grad', action='store_true', help="Remove the gradient mask within DEQ func.")

    add_deq_args(parser)
    return parser.parse_args()


def load_model(args):
    model = nn.DataParallel(DEQFlow(args), device_ids=args.gpus)
    checkpoint_path = "checkpoints/deq-flow-H-things-test-3x.pth"
    model.load_state_dict(torch.load(checkpoint_path), strict=False)

    model.cuda()
    model.eval()
    return model


if __name__ == "__main__":
    args = parse()

    nusc_config_path = "nuscenes_config.ini"
    nusc = NuScenesDataset(nusc_config_path)

    model = load_model(args)

    for scene in nusc.scenes:
        for sample_idx, sample in enumerate(scene.samples[:-1]):
            cam1 = scene.samples[sample_idx].sensors["CAM_FRONT"]
            cam2 = scene.samples[sample_idx + 1].sensors["CAM_FRONT"]

            img1 = cv2.imread(cam1.full_filename)
            img2 = cv2.imread(cam2.full_filename)
            img1 = cv2.resize(img1, (img1.shape[1] // 3, img1.shape[0] // 3))
            img2 = cv2.resize(img2, (img2.shape[1] // 3, img2.shape[0] // 3))
            img1_t = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2_t = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img1_t = np.transpose(img1_t, (2, 0, 1))
            img2_t = np.transpose(img2_t, (2, 0, 1))

            img1_t = torch.from_numpy(img1_t).float().cuda()
            img2_t = torch.from_numpy(img2_t).float().cuda()

            padder = InputPadder(img1_t.shape)

            img1_t, img2_t = padder.pad(img1_t[None].cuda(), img2_t[None].cuda())

            _, flow_pr, _ = model(img1_t, img2_t)
            
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().detach().numpy()
            img_flow = flow_viz.flow_to_image(flow)
            img_flow = cv2.cvtColor(img_flow, cv2.COLOR_RGB2BGR)
            img1 = cv2.resize(img1, (img1.shape[1], img1.shape[0]))
            img2 = cv2.resize(img2, (img2.shape[1], img2.shape[0]))
            img_flow = cv2.resize(img_flow, (img_flow.shape[1], img_flow.shape[0]))

            total_img = np.hstack((img1, img_flow, img2))
            cv2.imshow("flow", total_img)
            cv2.waitKey(100)

