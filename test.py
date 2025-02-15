# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import os
import argparse
import torch as th
import torch.nn.functional as F
import time
import conf_mgt
from utils import yamlread, str2bool
from guided_diffusion import dist_util

# Workaround
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass


from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)  # noqa: E402

def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample


def main(opt, conf: conf_mgt.Default_Conf):

    print("Start", conf['name'])

    device = dist_util.dev(conf.get('device'))


    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(
            opt.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    show_progress = opt.show_progress

    if conf.classifier_scale > 0 and conf.classifier_path:
        print("loading classifier...")
        classifier = create_classifier(
            **select_args(conf, classifier_defaults().keys()))
        classifier.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(
                conf.classifier_path), map_location="cpu")
        )

        classifier.to(device)
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        def cond_fn(x, t, y=None, gt=None, **kwargs):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return th.autograd.grad(selected.sum(), x_in)[0] * conf.classifier_scale
    else:
        cond_fn = None

    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return model(x, t, y if conf.class_cond else None, gt=gt)

    print("sampling...")

    model_kwargs = {}

    model_kwargs["gt"] = batch['gt']  # gt -> ori-image (targets remained to be removed)
    # read original image in: torch.size[1,3,w,h]

    gt_keep_mask = batch['gt_keep_mask'] # gt_keep_mask -> image mask (?)
    if gt_keep_mask is not None:
        model_kwargs['gt_keep_mask'] = gt_keep_mask

    batch_size = model_kwargs["gt"].shape[0]

    if conf.cond_y is not None:
        classes = th.ones(batch_size, dtype=th.long, device=device)
        model_kwargs["y"] = classes * conf.cond_y
    else:
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(batch_size,), device=device
        )
        model_kwargs["y"] = classes

    sample_fn = (
        diffusion.p_sample_loop if not opt.use_ddim else diffusion.ddim_sample_loop
    )


    result = sample_fn(
        model_fn,
        (batch_size, 3, opt.H, opt.W),
        clip_denoised=opt.clip_denoised,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn,
        device=device,
        progress=show_progress,
        return_all=True,
        conf=conf
    )
    srs = toU8(result['sample'])
    gts = toU8(result['gt'])
    lrs = toU8(result['gt'] * model_kwargs['gt_keep_mask'] + (-1) *
               th.ones_like(result['gt']) * (1 - model_kwargs['gt_keep_mask']))

    gt_keep_masks = toU8((model_kwargs['gt_keep_mask'] * 2 - 1))

    conf.eval_imswrite(
        srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gt_keep_masks,
        img_names=opt.out_name, dset='tmp', name='gt-test', verify_same=False)

    print("sampling complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    parser.add_argument('--out_name', type=str, required=False, default="gt-test")
    parser.add_argument('--show_progress', type=str2bool, required=False, default=True)
    parser.add_argument('--use_ddim', type=str2bool, required=False, default=False)
    parser.add_argument('--clip_denoised', type=str2bool, required=False, default=True)
    parser.add_argument('--model_path', type=str, required=False, default="/root/autodl-tmp/data/pretrained/celeba256_250000.pt")
    parser.add_argument('--gt_path', type=str, required=False,
                        default="/root/autodl-tmp/data/datasets/gts/face")
    parser.add_argument('--mask_path', type=str, required=False,
                        default="/root/autodl-tmp/data/datasets/gt_keep_masks/face")
    parser.add_argument('--W', type=int, required=False, default=512)
    parser.add_argument('--H', type=int, required=False, default=512)

    opt = parser.parse_args()

    conf_arg = conf_mgt.conf_base.Default_Conf(opt)
    conf_arg.update(yamlread(opt.conf_path))
    main(opt, conf_arg)
