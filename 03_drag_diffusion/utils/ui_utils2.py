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

import cv2
import pickle
import copy
import numpy as np
from copy import deepcopy
from types import SimpleNamespace
from skimage import morphology

import datetime
import PIL
from PIL import Image
from PIL.ImageOps import exif_transpose
from einops import rearrange
from collections import Counter
import torch
import torch.nn.functional as F

from diffusers import DDIMScheduler, AutoencoderKL, DPMSolverMultistepScheduler
from diffusers.models.embeddings import ImageProjection
from drag_pipeline import DragPipeline

from torchvision.utils import save_image
from pytorch_lightning import seed_everything

from .drag_utils2 import drag_diffusion_update
from .lora_utils import train_lora
from .attn_utils import register_attention_editor_diffusers, MutualSelfAttentionControl
from .freeu_utils import register_free_upblock2d, register_free_crossattn_upblock2d


def get_points_new(img,
               sel_pix):
    img_copy = img.copy()
    points = []
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            cv2.circle(img_copy, tuple(point), 10, (255, 0, 0), -1)
        else:
            cv2.circle(img_copy, tuple(point), 10, (0, 0, 255), -1)
        points.append(tuple(point))
        if len(points) == 2:
            cv2.arrowedLine(img_copy, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
            points = []
    return img_copy if isinstance(img_copy, np.ndarray) else np.array(img_copy)

def mask_image(image,
               mask,
               color=[255,0,0],
               alpha=0.5):
    """ Overlay mask on image for visualization purpose. 
    Args:
        image (H, W, 3) or (H, W): input image
        mask (H, W): mask to be overlaid
        color: the color of overlaid mask
        alpha: the transparency of the mask
    """
    out = deepcopy(image)
    img = deepcopy(image)
    img[mask == 1] = color
    out = cv2.addWeighted(img, alpha, out, 1-alpha, 0, out)
    return out

def store_mask(image, mask, mask_target, length=512):
    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        mask_target = np.uint8(mask_target > 0)
        masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = image.copy()
    return masked_img, mask, mask_target

def store_img(image, length=512):
    height,width, _ = image.shape
    image = Image.fromarray(image)
    image = exif_transpose(image)
    image = np.array(image)
    if height > width:
        image = cv2.resize(image, (length, int(length*height/width)))
    else:
        image = cv2.resize(image, (int(length*width/height), length))
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    return image, binary_image

def preprocess_image(image,
                     device,
                     dtype=torch.float32):
    image = torch.from_numpy(image).float() / 127.5 - 1 # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device, dtype)
    return image

def run_drag(source_image,
             mask,
             prompt,
             points,
             ref_points,
             inversion_strength,
             lam,
             latent_lr,
             n_pix_step,
             model_path,
             vae_path,
             lora_path,
             start_step,
             start_layer,
             save_dir,
             masklist
    ):
    # initialize model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False, steps_offset=1)
    model = DragPipeline.from_pretrained(model_path, scheduler=scheduler, torch_dtype=torch.float16)
    # call this function to override unet forward function,
    # so that intermediate features are returned after forward
    model.modify_unet_forward()
    model.to(device)
    # set vae
    if vae_path != "default":
        model.vae = AutoencoderKL.from_pretrained(
            vae_path
        ).to(model.vae.device, model.vae.dtype)

    # off load model to cpu, which save some memory.
    model.enable_model_cpu_offload()

    # initialize parameters
    seed = 915
    seed_everything(seed)

    args = SimpleNamespace()
    args.prompt = prompt
    args.points = points
    args.n_inference_step = 50
    args.n_actual_inference_step = round(inversion_strength * args.n_inference_step)
    args.guidance_scale = 1.0

    args.unet_feature_idx = [2]

    args.r_m = 1
    args.r_p = 3
    args.lam = lam

    args.lr = latent_lr
    args.n_pix_step = n_pix_step

    full_h, full_w = source_image.shape[:2]
    args.sup_res_h = int(0.5*full_h)
    args.sup_res_w = int(0.5*full_w)

    print(args)

    source_image = preprocess_image(source_image, device, dtype=torch.float16)

    mask = torch.from_numpy(mask).float() / 255.
    mask[mask > 0.0] = 1.0
    mask = rearrange(mask, "h w -> 1 1 h w").cuda()
    mask = F.interpolate(mask, (args.sup_res_h, args.sup_res_w), mode="nearest")
    
    handle_points = []
    target_points = []
    for idx, point in enumerate(points):
        cur_point = torch.tensor([point[1]/full_h*args.sup_res_h, point[0]/full_w*args.sup_res_w])
        cur_point = torch.round(cur_point)
        if idx % 2 == 0:
            handle_points.append(cur_point)
        else:
            target_points.append(cur_point)
    print('handle points:', handle_points)
    print('target points:', target_points)
    ref_pointsT = torch.tensor([ref_points[1]/full_h*args.sup_res_h, ref_points[0]/full_w*args.sup_res_w])
    ref_pointsT = torch.round(ref_pointsT)
    # set lora
    if lora_path == "":
        print("applying default parameters")
        model.unet.set_default_attn_processor()
    else:
        print("applying lora: " + lora_path)
        model.unet.load_attn_procs(lora_path)

    # obtain text embeddings
    text_embeddings = model.get_text_embeddings(prompt)

    # invert the source image
    # the latent code resolution is too small, only 64*64
    invert_code = model.invert(source_image,
                               prompt,
                               encoder_hidden_states=text_embeddings,
                               guidance_scale=args.guidance_scale,
                               num_inference_steps=args.n_inference_step,
                               num_actual_inference_steps=args.n_actual_inference_step)

    # empty cache to save memory
    torch.cuda.empty_cache()

    init_code = invert_code
    init_code_orig = deepcopy(init_code)
    model.scheduler.set_timesteps(args.n_inference_step)
    t = model.scheduler.timesteps[args.n_inference_step - args.n_actual_inference_step]

    # feature shape: [1280,16,16], [1280,32,32], [640,64,64], [320,64,64]
    # convert dtype to float for optimization
    init_code = init_code.float()
    text_embeddings = text_embeddings.float()
    model.unet = model.unet.float()

    updated_init_code,init_code_list = drag_diffusion_update(
        model,
        masklist,
        source_image,
        init_code,
        text_embeddings,
        t,
        handle_points,
        target_points,
        ref_pointsT, 
        mask,
        args)

    init_code_list = [i.half() for i in init_code_list]
    model_g = copy.deepcopy(model)
    updated_init_code = updated_init_code.half()
    text_embeddings = text_embeddings.half()
    model.unet = model.unet.half()

    # empty cache to save memory
    torch.cuda.empty_cache()

    # hijack the attention module
    # inject the reference branch to guide the generation
    editor = MutualSelfAttentionControl(start_step=start_step,
                                        start_layer=start_layer,
                                        total_steps=args.n_inference_step,
                                        guidance_scale=args.guidance_scale)
    if lora_path == "":
        register_attention_editor_diffusers(model, editor, attn_processor='attn_proc')
    else:
        register_attention_editor_diffusers(model, editor, attn_processor='lora_attn_proc')

    latents=torch.cat([init_code_orig, updated_init_code], dim=0)

    gen_image = model(
        prompt=args.prompt,
        encoder_hidden_states=torch.cat([text_embeddings] * 2, dim=0),
        batch_size=2,
        latents=torch.cat([init_code_orig, updated_init_code], dim=0),
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.n_inference_step,
        num_actual_inference_steps=args.n_actual_inference_step
    )[1].unsqueeze(dim=0)


    # resize gen_image into the size of source_image
    # we do this because sh1ape of gen_image will be rounded to multipliers of 8
    gen_image = F.interpolate(gen_image, (full_h, full_w), mode='bilinear')

    out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    out_image = (out_image * 255).astype(np.uint8)
    
    gray_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2GRAY)
    _, out_image_binary = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    return out_image, gray_image



# ------------------------------------------------------
#               Choosing useful points                 |
# ------------------------------------------------------

def longest_black_pixel_radio(image, A, B):
    """
    Calculate the length of the longest consecutive black pixel streak
    on the line between point A and point B in the given image.
    
    Parameters:
    - image: The grayscale image (numpy array)
    - A: Starting point (x1, y1)
    - B: Ending point (x2, y2)
    
    Returns:
    - max_black_length: The length of the longest consecutive black pixel streak
    """
    # Generate the coordinates of all the pixels on the line from A to B using the Bresenham algorithm
    image2 = copy.deepcopy(image)
    A = tuple(A)
    B = tuple(B)
    line_pixels = cv2.line(image2, A, B, 1, 1)
    line_indices = np.where(line_pixels == 1)

    # Extract the pixel values along the line
    line_pixel_values = image[line_indices]

    # Initialize variables
    max_black_length = 0
    current_black_length = 0

    # Iterate through the pixel values to find the longest streak of black pixels
    for pixel in line_pixel_values:
        if pixel == 0:  # If the pixel is black
            current_black_length += 1
            max_black_length = max(max_black_length, current_black_length)
        else:  # If the pixel is not black
            current_black_length = 0

    radio = int(max_black_length / len(line_pixel_values) * 100)
    return radio


def choose_handle_points(image, edge_points, cross_points, t):
    h, w = image.shape
    T_ref = t * min(h, w) / 512
    # cross-->1, edge-->0
    flag_xy = []
    step1_points = []
    data = []
    for index, candidate_point in enumerate(edge_points):
        # print("****************************")
        print("\nCurrent edge point:", candidate_point)
        step1_points = []
        flag_xy = []
        # obtain step_points1 & flag_xy
        #------------------------------
        result_cross_x = [candidate_point[0] - x for x, _ in cross_points]
        result_cross_y = [candidate_point[1] - y for _, y in cross_points]
        edge_points_else = [item for item in edge_points if item != candidate_point]
        result_edge_x = [candidate_point[0] - x for x, _ in edge_points_else]
        result_edge_y = [candidate_point[1] - y for _, y in edge_points_else]
        result_cross_x = [abs(x) for x in result_cross_x]
        result_cross_y = [abs(x) for x in result_cross_y]
        result_edge_x = [abs(x) for x in result_edge_x]
        result_edge_y = [abs(x) for x in result_edge_y]
        #  result_cross_x
        indices = [index for index, value in enumerate(result_cross_x) if value < T_ref]
        step1_points.extend([cross_points[i] for i in indices])
        flag_xy.extend([1] * len(indices))

        # result_cross_y
        indices = [index for index, value in enumerate(result_cross_y) if value < T_ref]
        step1_points.extend([cross_points[i] for i in indices])
        flag_xy.extend([1] * len(indices))

        # result_edge_x
        indices = [index for index, value in enumerate(result_edge_x) if value < T_ref]
        step1_points.extend([edge_points_else[i] for i in indices])
        flag_xy.extend([0] * len(indices))

        # result_edge_y
        indices = [index for index, value in enumerate(result_edge_y) if value < T_ref]
        step1_points.extend([edge_points_else[i] for i in indices])
        flag_xy.extend([0] * len(indices))


        # step2_points
        #-------------
        score_list = np.ones(len(step1_points))
        if len(step1_points) == 0:
            print("NO reference points!!!")
            continue
        else:
            radio_list = []
            for index_step1, step1_p in enumerate(step1_points):
                radio = longest_black_pixel_radio(image, candidate_point, step1_p)  
                if radio > 65:
                    radio_list.append(1)
                    score_list[index_step1] += 0
                else:
                    radio_list.append(0)
                    score_list[index_step1] += 2

                if flag_xy[index_step1] == 0:
                    score_list[index_step1] += 1
                else:
                    score_list[index_step1] += 0
            print("Score list:", score_list)

            max_index = np.argmax(score_list)
            indices = np.where(score_list==score_list[max_index])
            result = [step1_points[i] for i in indices[0]]
            sorted_data = sorted(result, key=lambda x: (x[1], x[0]))
            # score_index = [i for i, x in enumerate(score_list) if x == 3]
            # print("refrence point", list(map(lambda i: step1_points[i], score_index)))
            # return list(map(lambda i: step1_points[i], score_index))
            print("Reference point", sorted_data[0])

        data.append((candidate_point, sorted_data[0], score_list[max_index]))
    return data


def count_continous_black(image, ref, target_index, direction):
    count = 0
    x = ref[0] 
    y = ref[1]
    if target_index == 0:
        while 0 <= x < image.shape[1]:
            if image[ref[1], x] == 0:
                count += 1
                x += direction
            else:
                break
    else:
        while 0 <= y < image.shape[0]:
            if image[y, ref[0]] == 0:
                count += 1
                y += direction
            else:
                break
    return count

def relocate_refpoints(image, imagethin_binary, handle_points, ref_points, target_index):
    circle_mask = np.zeros_like(image)

    # radius = 80
    h, w = image.shape
    radius = int(50.0 * min(h, w) / 512)
    cv2.circle(circle_mask, handle_points, radius, (255,), thickness=3)
    black_pixels = (imagethin_binary == 0)
    circle_pixels = (circle_mask == 255)
    intersection = np.logical_and(black_pixels, circle_pixels)
    intersection_points = np.argwhere(intersection)
    if len(intersection_points) == 0:
        return ref_points
    average_x = int(np.mean(intersection_points[:, 1]))
    average_y = int(np.mean(intersection_points[:, 0]))
    # # # calculate target points
    slope_line = (int(handle_points[0]) - average_x, int(handle_points[1]) - average_y)
    ref_points = list(ref_points)
    # relocate ref_points when score is 3
    if target_index == 0:
        if slope_line[0] < 0:
            left_count = count_continous_black(image, ref_points, target_index, -1)
            ref_points[0] -= left_count
        else:
            right_count = count_continous_black(image, ref_points, target_index, 1)
            ref_points[0] += right_count
    else:
        if slope_line[1] < 0:
            up_count = count_continous_black(image, ref_points, target_index, -1)
            ref_points[1] -= up_count
        else:
            down_count = count_continous_black(image, ref_points, target_index, 1)
            ref_points[1] += down_count
    # print("Corrected ref_points:", ref_points)
    return ref_points



def find_furthest_point(img, start_point, unit_vector):
    height, width = img.shape[:2]
    current_point = start_point
    dx = round(2 * unit_vector[0])
    dy = round(2 * unit_vector[1])
    if img[current_point[1], current_point[0]] != 0: 
        print("The start point is not a black pixel")
        return start_point

    while True:
        next_point = (current_point[0] + dx, current_point[1] + dy)
        if not (0 <= next_point[0] < width and 0 <= next_point[1] < height):
            break
        if img[next_point[1], next_point[0]] != 0:
            dx = round(1 * np.sign(unit_vector[0]))
            dy = round(1 * np.sign(unit_vector[1]))
            next_point = (current_point[0] + dx, current_point[1] + dy)
            if img[next_point[1], next_point[0]] == 0:
                current_point = next_point
                break
            else:
                break
        current_point = next_point
    return current_point


def relocate_edgepoints(points, image, imagethin):
    x = points[0]
    y = points[1]
    center = (int(x), int(y)) 
    circle_mask = np.zeros_like(image)
    T = 90
    h, w = image.shape
    radius = int(T * min(h, w) / 512)
    cv2.circle(circle_mask, center, radius, (255,), thickness=3)
    black_pixels = (imagethin == 0)
    circle_pixels = (circle_mask == 255)
    intersection = np.logical_and(black_pixels, circle_pixels)
    intersection_points = np.argwhere(intersection)
    if len(intersection_points) == 0:
        return tuple(points)

    average_x = int(np.mean(intersection_points[:, 1]))
    average_y = int(np.mean(intersection_points[:, 0]))

    slope_line = (int(x) - average_x, int(y) - average_y)
    length_line = np.sqrt(slope_line[0]**2 + slope_line[1]**2)
    unit_vector = (slope_line[0] / length_line, slope_line[1] / length_line)

    furthest_point = find_furthest_point(image, points, unit_vector)

    return furthest_point

def find_closetpoint(image, point):
    image = image[:, :, 0]
    min_dist = float('inf')
    closest_black_pixel = point
    for y in range(point[1]-20, point[1]+20):
        for x in range(point[0]-20, point[0]+20):
            if image[y, x] == 0:
                dist = np.sqrt((x - point[0])**2 + (y - point[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_black_pixel = (x, y)
    return closest_black_pixel

def choose_target_points(image, handle_points, ref_points, name, target_index, shift_d, w):
    img_path = "./data/direction/"
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_binary = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY)
    image_binary = np.stack([image_binary] * 3, axis=-1)
    x = handle_points[0]
    y = handle_points[1]
    center = (int(x), int(y)) 
    handle_points = find_closetpoint(image, center)
    circle_mask = np.zeros_like(image)
    radius = 60
    cv2.circle(circle_mask, center, radius, (255,), thickness=6)
    black_pixels = (image_binary == 0)
    circle_pixels = (circle_mask == 255)
    intersection = np.logical_and(black_pixels, circle_pixels)
    intersection_points = np.argwhere(intersection)
    average_x = int(np.mean(intersection_points[:, 1]))
    average_y = int(np.mean(intersection_points[:, 0]))
    average_coordinate = (average_x, average_y)
    cv2.line(image, average_coordinate, center, (0, 0, 255), thickness=5)

    d_cofficient = 0
    slope_line = (int(x) - average_x, int(y) - average_y)
    if target_index == 0:
        if w == '0':
            if handle_points[0] > ref_points[0]:
                if slope_line[0] > 0:
                    d_cofficient = -1
                else:
                    d_cofficient = 1
            else:
                if slope_line[0] < 0:
                    d_cofficient = -1
                else:
                    d_cofficient = 1
        else:
            if handle_points[0] > ref_points[0]:
                if slope_line[0] > 0:
                    d_cofficient = 1
                else:
                    d_cofficient = -1
            else:
                if slope_line[0] < 0:
                    d_cofficient = 1
                else:
                    d_cofficient = -1

    else:
        if w == '0':
            d_cofficient = np.sign(slope_line[1])
        else:
            if slope_line[1] > 0:
                slope = slope_line[0] / slope_line[1]
            else:
                slope = 0
            if slope == 0:
                if slope_line[1] <= 0:
                    d_cofficient = 1
                else:
                    d_cofficient = -1
            elif slope > 0:
                if slope_line[0] < 0:
                    d_cofficient = 1
                else:
                    d_cofficient = -1
            else:
                if slope_line[0] < 0:
                    d_cofficient = -1
                else:
                    d_cofficient = 1

    shift_d = d_cofficient * np.abs(shift_d)
    length_line = np.sqrt(slope_line[0]**2 + slope_line[1]**2)
    unit_vector = (slope_line[0] / length_line, slope_line[1] / length_line)
    target_points = (round(handle_points[0] + unit_vector[0] * shift_d), round(handle_points[1] + unit_vector[1] * shift_d))
    target_points = tuple(max(1, x) for x in target_points)
    new_x = min(max(1, target_points[0]), image.shape[1])
    new_y = min(max(1, target_points[1]), image.shape[0])
    target_points = (new_x, new_y)
    cv2.circle(image, target_points, radius=2, color=(255, 0, 0), thickness=5)
    cv2.circle(image, tuple(handle_points), radius=2, color=(0, 255, 0), thickness=5)
    cv2.imwrite(img_path + "line_" + name + ".png", image)
    return target_points, handle_points, slope_line

def calculateIndex(pixel_string, T):
    pixel_string1 = pixel_string[1:]
    pixel_string1 = np.append(pixel_string1, 254)
    tmp = pixel_string - pixel_string1
    index_tmp = np.where(tmp == 255)[0]
    filtered_values = index_tmp[index_tmp <= T]
    if len(filtered_values) == 0:
        index1 = T
    else:
        index1 = np.max(filtered_values)
    index_tmp = np.where(tmp == -255)[0]
    filtered_values = index_tmp[index_tmp >= T]
    if len(filtered_values) == 0:
        index2 = T
    else:
        index2 = np.min(filtered_values)
    return index1, index2

def get_perpendicular_direction(dir):
    """Return the perpendicular axis-aligned direction."""
    if dir == (0, 1) or dir == (0, -1):
        return (1, 0)
    elif dir == (1, 0) or dir == (-1, 0):
        return (0, 1)
    else:
        raise ValueError("only horizontal and vertical directions are supported")
    
def count_black_pixels(matrix, start_point, direction):
    """Count black pixels along the perpendicular direction while walking from a start point."""
    x, y = start_point
    dx, dy = direction
    perpendicular_direction = get_perpendicular_direction(direction)
    px, py = perpendicular_direction
    S = []

    while 0 <= x < matrix.shape[1] and 0 <= y < matrix.shape[0] and matrix[y, x, 0] == 0:
        count = 0
        px_pos, py_pos = x, y
        while 0 <= px_pos < matrix.shape[0] and 0 <= py_pos < matrix.shape[1] and matrix[py_pos, px_pos, 0] == 0:
            count += 1
            px_pos += px
            py_pos += py
        S.append((count, (x, y)))
        
        x += dx
        y += dy

    return S

def most_frequent_value_coords(S):
    counts = [count for count, _ in S]
    count = Counter(counts)
    
    if all(value == 1 for value in count.values()):
        avg_value = sum(counts) / len(counts)
        avg_coords = [coord for value, coord in S if value == round(avg_value)]
        return avg_coords
    
    most_common_value = count.most_common(1)[0][0]
    most_common_coords = [coord for value, coord in S if value == most_common_value]
    return most_common_coords

def choose_mask(image, handle_points, target_points, target_index, name, slope_line):
    img_path = "./data/mask/"

    height, width, _ = image.shape
    margin = 10
    x1 = handle_points[0]
    y1 = handle_points[1]
    x2 = target_points[0]
    y2 = target_points[1]
    index1All = []
    index2All = []

    add_margin = 10
    if target_index == 0:
        T = handle_points[1]
        min_x = max(min(x1, x2) - margin, 0)
        max_x = min(max(x1, x2) + margin, width - 1)
        pixel_string = image[:, :, 0][:, min_x: max_x]
        columns_with_not_all_white = np.where(pixel_string[handle_points[1], :] == 0)[0]
        for i in columns_with_not_all_white:
            pixels = pixel_string[:, i]
            index1, index2 = calculateIndex(pixels, T)
            index1All.append(index1)
            index2All.append(index2)
        if len(index1All) > 0:
            up_index = min(index1All)
        else:
            up_index = y1
        if len(index2All) > 0:
            down_index = max(index2All)
        else:
            down_index = y2
        min_y = max(min(y1, y2, up_index) - margin - add_margin, 0)
        max_y = min(max(y1, y2, down_index) + margin + add_margin, height - 1)
    else:
        T = handle_points[0]
        min_y = max(min(y1, y2) - margin, 0)
        max_y = min(max(y1, y2) + margin, height - 1)
        pixel_string = image[:, :, 0][min_y:max_y, :]
        rows_with_not_all_white = np.where(pixel_string[:, handle_points[0]] == 0)[0]
        for i in rows_with_not_all_white:
            pixels = pixel_string[i, :]
            index1, index2 = calculateIndex(pixels, T)
            index1All.append(index1)
            index2All.append(index2)
        if len(index1All) > 0:
            left_index = min(index1All)
        else:
            left_index = x1
        if len(index2All) > 0:
            right_index = max(index2All)
        else:
            right_index = x2

        min_x = max(min(x1, x2, left_index) - margin - add_margin, 0)
        max_x = min(max(x1, x2, right_index) + margin + add_margin, width - 1)

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[min_y: max_y, min_x: max_x] = 1
    dx = target_points[0] - handle_points[0]
    dy = target_points[1] - handle_points[1]
    new_min_x = min_x + dx
    new_max_x = max_x + dx
    new_min_y = min_y + dy
    new_max_y = max_y + dy
    mask_target = np.zeros((height, width), dtype=np.uint8)
    mask_target[new_min_y: new_max_y, new_min_x: new_max_x] = 1
    masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    cv2.imwrite(img_path + "mask_" + name + ".jpg", masked_img)
    masklist = [min_y, max_y, min_x, max_x]
    return mask * 255, mask_target * 255, masklist

def choose_maskReplace(image, handle_points, target_points, target_index, name, slope_line):
    height, width, _ = image.shape
    margin = 2
    x1 = handle_points[0]
    y1 = handle_points[1]
    x2 = target_points[0]
    y2 = target_points[1]
    index1All = []
    index2All = []
    direction = np.argmin(np.abs(np.array(handle_points) - np.array(target_points)))
    if target_index == 0:
        if slope_line[0] > 0:
            direction = (-1, 0)
        else:
            direction = (1, 0)
        if target_points[0] - handle_points[0] < 0:
            min_x = max(min(x1, x2) - margin, 0)
            max_x = min(max(x1, x2) + margin + 10, width - 1)
        else:
            min_x = max(min(x1, x2) - margin - 10, 0)
            max_x = min(max(x1, x2) + margin, width - 1)
    else:
        if slope_line[1] > 0:
            direction = (0, -1)
        else:
            direction = (0, 1)
        if target_points[1] - handle_points[1] < 0:
            min_y = max(min(y1, y2) - margin, 0)
            max_y = min(max(y1, y2) + margin + 10, height - 1)
        else:
            min_y = max(min(y1, y2) - margin - 10, 0)
            max_y = min(max(y1, y2) + margin, height - 1)

    add_margin = 20
    if target_index == 0:
        T = handle_points[1]
        pixel_string = image[:, :, 0][:, min_x: max_x]
        columns_with_not_all_white = np.where(pixel_string[handle_points[1], :] == 0)[0]
        for i in columns_with_not_all_white:
            pixels = pixel_string[:, i]
            index1, index2 = calculateIndex(pixels, T)
            index1All.append(index1)
            index2All.append(index2)
        if len(index1All) > 0:
            up_index = min(index1All)
        else:
            up_index = y1
        if len(index2All) > 0:
            down_index = max(index2All)
        else:
            down_index = y2
        min_y = max(min(y1, y2, up_index) - margin - add_margin, 0)
        max_y = min(max(y1, y2, down_index) + margin + add_margin, height - 1)
    else:
        T = handle_points[0]
        pixel_string = image[:, :, 0][min_y:max_y, :]
        rows_with_not_all_white = np.where(pixel_string[:, handle_points[0]] == 0)[0]
        for i in rows_with_not_all_white:
            pixels = pixel_string[i, :]
            index1, index2 = calculateIndex(pixels, T)
            index1All.append(index1)
            index2All.append(index2)
        if len(index1All) > 0:
            left_index = min(index1All)
        else:
            left_index = x1
        if len(index2All) > 0:
            right_index = max(index2All)
        else:
            right_index = x2

        min_x = max(min(x1, x2, left_index) - margin - add_margin, 0)
        max_x = min(max(x1, x2, right_index) + margin + add_margin, width - 1)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[min_y: max_y, min_x: max_x] = 1
    return mask * 255

def extract_skeleton(image):
    length = 512
    h, w, _ = image.shape
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = ~gray_image
    _, image_binary = cv2.threshold(gray_image, 128, 1, cv2.THRESH_BINARY)
    iThinBool = morphology.skeletonize(image_binary)
    iThin = np.array(iThinBool, dtype=int)
    iThin = 1 - iThin
    if w < h:
        img = cv2.resize(np.uint8(iThin*255), (512, int(length*h/w)))
    else:
        img = cv2.resize(np.uint8(iThin*255), (int(length*w/h), 512))

    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    return img



def extract_skeleton_attack(image, kernelSize):
    length = 512
    h, w, _ = image.shape
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = ~gray_image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernelSize, kernelSize))
    dilation = cv2.dilate(gray_image, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernelSize- 3, kernelSize-3))
    gray_image = cv2.erode(dilation, kernel)
    _, image_binary = cv2.threshold(gray_image, 128, 1, cv2.THRESH_BINARY)
    iThinBool = morphology.skeletonize(image_binary)
    iThin = np.array(iThinBool, dtype=int)
    iThin = 1 - iThin
    if w < h:
        img = cv2.resize(np.uint8(iThin*255), (512, int(length*h/w)))
    else:
        img = cv2.resize(np.uint8(iThin*255), (int(length*w/h), 512))

    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    return img
