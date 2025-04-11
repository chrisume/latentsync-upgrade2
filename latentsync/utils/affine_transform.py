# Adapted from https://github.com/guanjz20/StyleSync/blob/main/utils.py

import numpy as np
import cv2
import torch
from einops import rearrange
import kornia

device = "cuda:0"
dtype = torch.float16

def transformation_from_points(points1, points0, smooth=True, p_bias=None):
    points2 = np.array(points0)
    points2 = points2.astype(np.float64)
    points1 = points1.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(np.matmul(points1.T, points2))
    R = (np.matmul(U, Vt)).T
    sR = (s2 / s1) * R
    T = c2.reshape(2, 1) - (s2 / s1) * np.matmul(R, c1.reshape(2, 1))
    M = np.concatenate((sR, T), axis=1)
    if smooth:
        bias = points2[2] - points1[2]
        if p_bias is None:
            p_bias = bias
        else:
            bias = p_bias * 0.2 + bias * 0.8
        p_bias = bias
        M[:, 2] = M[:, 2] + bias
    return M, p_bias

def transformation_from_points_torch(points1, points0, smooth=True, p_bias=None):  

    if isinstance(points0, np.ndarray):  
        points2 = torch.tensor(points0, device=device, dtype=torch.float32)
    else:  
        points2 = points0.clone()  
        
    if isinstance(points1, np.ndarray):  
        points1_tensor = torch.tensor(points1, device=device, dtype=torch.float32)
    else:  
        points1_tensor = points1.clone()
    
    c1 = torch.mean(points1_tensor, dim=0)  
    c2 = torch.mean(points2, dim=0)  
    
    points1_centered = points1_tensor - c1  
    points2_centered = points2 - c2  
    
    s1 = torch.std(points1_centered)  
    s2 = torch.std(points2_centered)  
    
    points1_normalized = points1_centered / s1  
    points2_normalized = points2_centered / s2  
    
    covariance = torch.matmul(points1_normalized.T, points2_normalized)  
    U, S, V = torch.svd(covariance)  
    
    R = torch.matmul(V, U.T)  
    
    det = torch.det(R)  
    if det < 0:  
        V[:, -1] = -V[:, -1]  
        R = torch.matmul(V, U.T)  
    
    sR = (s2 / s1) * R  
    T = c2.reshape(2, 1) - (s2 / s1) * torch.matmul(R, c1.reshape(2, 1))  
    
    M = torch.cat((sR, T), dim=1)  
 
    if smooth:  
        bias = points2_normalized[2] - points1_normalized[2]
        if p_bias is None:  
            p_bias = bias
        else:
            bias = p_bias * 0.2 + bias * 0.8  
            
        p_bias = bias
        
        M[:, 2] = M[:, 2] + bias
    
    return M.cpu().numpy(), p_bias 

class AlignRestore(object):
    def __init__(self, align_points=3):
        if align_points == 3:
            self.upscale_factor = 1
            ratio = 2.8
            self.crop_ratio = (ratio, ratio)
            self.face_template = np.array([[19 - 2, 30 - 10], [56 + 2, 30 - 10], [37.5, 45 - 5]])
            self.face_template = self.face_template * ratio
            self.face_size = (int(75 * self.crop_ratio[0]), int(100 * self.crop_ratio[1]))
            self.p_bias = None
            self.fill_value = torch.tensor([127, 127, 127], dtype=dtype).to(device=device)
            self.mask = torch.ones((1, 1, self.face_size[1], self.face_size[0]), device=device, dtype=dtype)

    def process(self, img, lmk_align=None, smooth=True, align_points=3):
        # TODO: change this to gpu (When inference, the "process" is not use)
        aligned_face, affine_matrix = self.align_warp_face(img, lmk_align, smooth)
        restored_img = self.restore_img(img, aligned_face, affine_matrix)
        cv2.imwrite("restored.jpg", restored_img)
        cv2.imwrite("aligned.jpg", aligned_face)
        return aligned_face, restored_img

    def align_warp_face(self, img, lmks3, smooth=True, border_mode="constant"):
        affine_matrix, self.p_bias = transformation_from_points_torch(lmks3, self.face_template, smooth, self.p_bias)
        if border_mode == "constant":
            border_mode = cv2.BORDER_CONSTANT
        elif border_mode == "reflect101":
            border_mode = cv2.BORDER_REFLECT101
        elif border_mode == "reflect":
            border_mode = cv2.BORDER_REFLECT

        img_tensor = rearrange(torch.from_numpy(img).to(device=device, dtype=dtype), "h w c -> c h w")
        img_tensor = img_tensor.unsqueeze(0)
        affine_matrix_tensor = torch.from_numpy(affine_matrix).to(device=device, dtype=dtype).unsqueeze(0)

        crop_tensor = kornia.geometry.transform.warp_affine(img_tensor, affine_matrix_tensor, (self.face_size[1], self.face_size[0]), mode='bilinear', padding_mode='fill', fill_value=self.fill_value)
        cropped_face = rearrange(crop_tensor.squeeze(0), "c h w -> h w c").cpu().numpy().astype(np.uint8)
        return cropped_face, affine_matrix
    
    def align_warp_face2(self, img, landmark, border_mode="constant"):
        # TODO: change this process(When inference, the "align_warp_face2" is not use)
        affine_matrix = cv2.estimateAffinePartial2D(landmark, self.face_template)[0]
        if border_mode == "constant":
            border_mode = cv2.BORDER_CONSTANT
        elif border_mode == "reflect101":
            border_mode = cv2.BORDER_REFLECT101
        elif border_mode == "reflect":
            border_mode = cv2.BORDER_REFLECT
        cropped_face = cv2.warpAffine(
            img, affine_matrix, self.face_size, borderMode=border_mode, borderValue=(135, 133, 132)
        )
        return cropped_face, affine_matrix

    def restore_img(self, input_img, face, torch_affine_matrix, more_fast=False):
        h, w, _ = input_img.shape

        if type(torch_affine_matrix) == np.ndarray:
            torch_affine_matrix = torch.from_numpy(torch_affine_matrix).to(device=device, dtype=dtype).unsqueeze(0)

        torch_inverse_affine = kornia.geometry.transform.invert_affine_transform(torch_affine_matrix)
        # H, W, C -> C, H, W
        dsize = (h, w)
        face_tensor = rearrange(torch.from_numpy(face).to(device=device, dtype=dtype), "h w c -> c h w")
        
        inv_restored = kornia.geometry.transform.warp_affine(face_tensor.unsqueeze(0), torch_inverse_affine, dsize, mode='bilinear', padding_mode='fill', fill_value=self.fill_value).squeeze(0)

        if more_fast:
            input_tensor = rearrange(torch.from_numpy(input_img).to(device=device, dtype=dtype), "h w c -> c h w")
            inv_mask = kornia.geometry.transform.warp_affine(self.mask, torch_inverse_affine, (h, w), mode='bilinear', padding_mode='zeros').squeeze(0)  # [1,h_up,w_up]
            # TODO: in the future, we can use landmark_106 fast create better mask
            inv_soft_mask_3d = inv_mask.expand_as(inv_restored)

            tensor_img_back = inv_soft_mask_3d * inv_restored + (1 - inv_soft_mask_3d) * input_tensor

        else:
            input_tensor = rearrange(torch.from_numpy(input_img).to(device=device, dtype=dtype), "h w c -> c h w")
            inv_mask = kornia.geometry.transform.warp_affine(self.mask, torch_inverse_affine, (h, w), mode='bilinear', padding_mode='zeros')  # [1,1,h_up,w_up]
            
            inv_mask_erosion = kornia.morphology.erosion(inv_mask, torch.ones((int(2 * self.upscale_factor), int(2 * self.upscale_factor)), device=device, dtype=dtype))

            inv_mask_erosion_t = inv_mask_erosion.squeeze(0).expand_as(inv_restored)
            pasted_face = inv_mask_erosion_t * inv_restored
            total_face_area = torch.sum(inv_mask_erosion.float())
            w_edge = int(total_face_area**0.5) // 20
            erosion_radius = w_edge * 2
            inv_mask_center = kornia.morphology.erosion(inv_mask_erosion, torch.ones((erosion_radius, erosion_radius), device=device, dtype=dtype))
            blur_size = w_edge * 2 + 1
            sigma = 0.3*((blur_size-1)*0.5 - 1) + 0.8
            inv_soft_mask = kornia.filters.gaussian_blur2d(inv_mask_center, (blur_size, blur_size), (sigma, sigma)).squeeze(0)
            inv_soft_mask_3d = inv_soft_mask.expand_as(inv_restored)
            tensor_img_back = inv_soft_mask_3d * pasted_face + (1 - inv_soft_mask_3d) * input_tensor
            
        tensor_img_back = rearrange(tensor_img_back, "c h w -> h w c").contiguous().to(dtype=torch.uint8)
        img_back = tensor_img_back.cpu().numpy()
        return img_back


class laplacianSmooth:
    def __init__(self, smoothAlpha=0.3):
        self.smoothAlpha = smoothAlpha
        self.pts_last = None

    def smooth(self, pts_cur):
        if self.pts_last is None:
            self.pts_last = pts_cur.copy()
            return pts_cur.copy()
        x1 = min(pts_cur[:, 0])
        x2 = max(pts_cur[:, 0])
        y1 = min(pts_cur[:, 1])
        y2 = max(pts_cur[:, 1])
        width = x2 - x1
        pts_update = []
        for i in range(len(pts_cur)):
            x_new, y_new = pts_cur[i]
            x_old, y_old = self.pts_last[i]
            tmp = (x_new - x_old) ** 2 + (y_new - y_old) ** 2
            w = np.exp(-tmp / (width * self.smoothAlpha))
            x = x_old * w + x_new * (1 - w)
            y = y_old * w + y_new * (1 - w)
            pts_update.append([x, y])
        pts_update = np.array(pts_update)
        self.pts_last = pts_update.copy()

        return pts_update
