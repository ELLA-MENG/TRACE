# *************************************************************************
# Copyright (2023) Bytedance Inc.
#
# Copyright (2023) DragDiffusion Authors 
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
#     http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
# *************************************************************************

import copy
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from einops import rearrange

def point_tracking(F0,
                   F1,
                   handle_points,
                   handle_points_init,
                   args):
    with torch.no_grad():
        _, _, max_r, max_c = F0.shape
        for i in range(len(handle_points)):
            pi0, pi = handle_points_init[i], handle_points[i]
            f0 = F0[:, :, int(pi0[0]), int(pi0[1])]

            r1, r2 = max(0,int(pi[0])-args.r_p), min(max_r,int(pi[0])+args.r_p+1)
            c1, c2 = max(0,int(pi[1])-args.r_p), min(max_c,int(pi[1])+args.r_p+1)
            F1_neighbor = F1[:, :, r1:r2, c1:c2]
            all_dist = (f0.unsqueeze(dim=-1).unsqueeze(dim=-1) - F1_neighbor).abs().sum(dim=1)
            all_dist = all_dist.squeeze(dim=0)
            row, col = divmod(all_dist.argmin().item(), all_dist.shape[-1])
            # handle_points[i][0] = pi[0] - args.r_p + row
            # handle_points[i][1] = pi[1] - args.r_p + col
            handle_points[i][0] = r1 + row
            handle_points[i][1] = c1 + col
        return handle_points

def check_handle_reach_target(handle_points,
                              target_points):
    # dist = (torch.cat(handle_points,dim=0) - torch.cat(target_points,dim=0)).norm(dim=-1)
    all_dist = list(map(lambda p,q: (p-q).norm(), handle_points, target_points))
    # return (torch.tensor(all_dist) < 2.0).all()
    return (torch.tensor(all_dist) < 2).all()

# obtain the bilinear interpolated feature patch centered around (x, y) with radius r
def interpolate_feature_patch(feat,
                              y1,
                              y2,
                              x1,
                              x2):
    x1_floor = torch.floor(x1).long()
    x1_cell = x1_floor + 1
    dx = torch.floor(x2).long() - torch.floor(x1).long()

    y1_floor = torch.floor(y1).long()
    y1_cell = y1_floor + 1
    dy = torch.floor(y2).long() - torch.floor(y1).long()

    wa = (x1_cell.float() - x1) * (y1_cell.float() - y1)
    wb = (x1_cell.float() - x1) * (y1 - y1_floor.float())
    wc = (x1 - x1_floor.float()) * (y1_cell.float() - y1)
    wd = (x1 - x1_floor.float()) * (y1 - y1_floor.float())

    Ia = feat[:, :, y1_floor : y1_floor+dy, x1_floor : x1_floor+dx]
    Ib = feat[:, :, y1_cell : y1_cell+dy, x1_floor : x1_floor+dx]
    Ic = feat[:, :, y1_floor : y1_floor+dy, x1_cell : x1_cell+dx]
    Id = feat[:, :, y1_cell : y1_cell+dy, x1_cell : x1_cell+dx]

    return Ia * wa + Ib * wb + Ic * wc + Id * wd

def drag_diffusion_update(model,
                          masklist,
                          image,
                          init_code,
                          text_embeddings,
                          t,
                          handle_points,
                          target_points,
                          points_refT,
                          mask,
                          args):

    assert len(handle_points) == len(target_points), \
        "number of handle point must equals target points"
    if text_embeddings is None:
        text_embeddings = model.get_text_embeddings(args.prompt)

    # the init output feature of unet
    with torch.no_grad():
        unet_output, F0 = model.forward_unet_features(init_code, t,
            encoder_hidden_states=text_embeddings,
            layer_idx=args.unet_feature_idx, interp_res_h=args.sup_res_h, interp_res_w=args.sup_res_w)
        x_prev_0,_ = model.step(unet_output, t, init_code)
        # init_code_orig = copy.deepcopy(init_code)

    # prepare optimizable init_code and optimizer
    init_code_orig = copy.deepcopy(init_code)
    init_code.requires_grad_(True)
    optimizer = torch.optim.Adam([init_code], lr=args.lr)

    # prepare for point tracking and background regularization
    handle_points_init = copy.deepcopy(handle_points)
    interp_mask = F.interpolate(mask, (init_code.shape[2],init_code.shape[3]), mode='nearest')
    using_mask = interp_mask.sum() != 0.0

    tmp = handle_points_init[0].float()
    handle_pointsinitPix = torch.tensor([tmp[1] * 2, tmp[0] * 2])
    #########################
    # torch.cuda.empty_cache()
    
    # prepare amp scaler for mixed-precision training
    
    init_code_list = []
    scaler = torch.cuda.amp.GradScaler()
    stage2_flag = 0

    count_style = 0
    flag = 0
    args.n_pix_step = 100
    for step_idx in range(args.n_pix_step):
        print('step_idx',step_idx)
        ###############
        torch.cuda.empty_cache()
        # if stage2_flag == 20:
        #     break
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            unet_output, F1 = model.forward_unet_features(init_code, t,
                encoder_hidden_states=text_embeddings,
                layer_idx=args.unet_feature_idx, interp_res_h=args.sup_res_h, interp_res_w=args.sup_res_w)
            x_prev_updated,_ = model.step(unet_output, t, init_code)

            # do point tracking to update handle points before computing motion supervision loss
            if step_idx != 0:
                handle_points = point_tracking(F0, F1, handle_points, handle_points_init, args)
                print('new handle points', handle_points)

            # Calculate new mask
            handle_points2 = handle_points[0].float()
            handle_points_pix2 = torch.tensor([handle_points2[1] * 2, handle_points2[0] * 2])
            move_x = handle_points_pix2[0] - handle_pointsinitPix[0]
            move_y = handle_points_pix2[1] - handle_pointsinitPix[1]
            new_min_x = masklist[2] + move_x
            new_max_x = masklist[3] + move_x
            new_min_y = masklist[0] + move_y
            new_max_y = masklist[1] + move_y
            new_min_x = int(new_min_x.item())
            new_max_x = int(new_max_x.item())
            new_min_y = int(new_min_y.item())
            new_max_y = int(new_max_y.item())
            new_mask = np.zeros((image.shape[2], image.shape[3]), dtype=np.uint8)
            new_mask[new_min_y: new_max_y, new_min_x: new_max_x] = 1
            new_mask = torch.from_numpy(new_mask).float() / 255.
            new_mask[new_mask > 0.0] = 1.0
            new_mask = rearrange(new_mask, "h w -> 1 1 h w").cuda()
            new_mask = F.interpolate(new_mask, (args.sup_res_h, args.sup_res_w), mode="nearest")
            interp_mask2 = F.interpolate(new_mask, (init_code.shape[2],init_code.shape[3]), mode='nearest')
            # break if all handle points have reached the targets
            if check_handle_reach_target(handle_points, target_points):
                if stage2_flag == 0:
                    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$reach_target!!!$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n\n')
                    break
                    flag = 1

            loss = 0.0
            # if flag == 0:
            _, _, max_r, max_c = F0.shape
            if flag == 0:
                for i in range(len(handle_points)):
                    pi, ti = handle_points[i], target_points[i]
                    # skip if the distance between target and source is less than 1
                    if (ti - pi).norm() < 2.:
                        print('skip if the distance between target and source is less than 1')
                        continue
                    # jisuan qianjin de fangxiang
                    di = (ti - pi) / (ti - pi).norm()

                    # motion supervision
                    # with boundary protection
                    args.r_m = 1
                    r1, r2 = max(0,int(pi[0])-args.r_m), min(max_r,int(pi[0])+args.r_m+1)
                    c1, c2 = max(0,int(pi[1])-args.r_m), min(max_c,int(pi[1])+args.r_m+1)
                    f0_patch = F1[:,:,r1:r2, c1:c2].detach()
                    f1_patch = interpolate_feature_patch(F1,r1+di[0],r2+di[0],c1+di[1],c2+di[1])
                    # Compare with the source point.


                    loss_drag = ((2*args.r_m+1)**2)*F.l1_loss(f0_patch, f1_patch)
                    loss += loss_drag
                    print("loss_drag: ", loss_drag)

                # unmasked region must stay unchanged
                if using_mask:
                    loss_mask1 = args.lam * ((x_prev_updated-x_prev_0)*(1-interp_mask)).abs().sum()
                    loss_mask = args.lam * (x_prev_updated*interp_mask2 - x_prev_0*interp_mask).abs().sum()/1000
                    loss += loss_mask + loss_mask1


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return init_code,init_code_list
