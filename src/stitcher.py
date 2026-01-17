import numpy as np
import cv2
from typing import Tuple, Optional

class ImageStitcher:
    def __init__(self, focal_length=None):
        self.focal_length = focal_length

    # phep chieu tru
    def cylindrical_projection(self, img: np.ndarray, focal_length=None) -> np.ndarray:
        if focal_length is None:
            focal_length = self.focal_length
        if focal_length is None:
            h, w = img.shape[:2]
            focal_length = (w + h) / 2
        
        h, w = img.shape[:2]
        
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        theta = (x_coords - w/2) / focal_length
        h_val = (y_coords - h/2) / focal_length
        
        X = np.sin(theta)
        Y = h_val
        Z = np.cos(theta)
        
        x_proj = (focal_length * X / Z + w/2).astype(np.float32)
        y_proj = (focal_length * Y / Z + h/2).astype(np.float32)
        
        cylindrical = cv2.remap(img, x_proj, y_proj, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        return cylindrical

    def normalize_points(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        centroid = np.mean(pts, axis=0)
        shifted = pts - centroid
        mean_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))
        
        if mean_dist < 1e-10:
            scale = 1.0
        else:
            scale = np.sqrt(2) / mean_dist
        
        T = np.array([
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        normalized = (T @ np.hstack((pts, np.ones((len(pts), 1)))).T).T[:, :2]
        return normalized, T

    def compute_dlt(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
        if len(src_pts) < 4:
            raise ValueError("Can it nhat 4 cap diem de tinh homography.")
        
        if len(src_pts) != len(dst_pts):
            raise ValueError("So diem nguon va dich phai bang nhau.")
        
        src_norm, T_src = self.normalize_points(src_pts)
        dst_norm, T_dst = self.normalize_points(dst_pts)
        
        A = []
        for i in range(len(src_norm)):
            x, y = src_norm[i]
            u, v = dst_norm[i]
            A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
            A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
        
        A = np.array(A, dtype=np.float64)
        
        try:
            _, _, Vh = np.linalg.svd(A)
            H_norm = Vh[-1].reshape(3, 3)
            
            if abs(H_norm[2, 2]) < 1e-8:
                raise ValueError("Homography suy bien (H[2,2] xap xi 0) - co the do diem thang hang hoac nhieu lop.")
            
            H_norm = H_norm / H_norm[2, 2]
            H = np.linalg.inv(T_dst) @ H_norm @ T_src
            H = H / H[2, 2]
            return H
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"Khong the tinh SVD: {str(e)}")

    def _ransac_homography(self, src_pts: np.ndarray, dst_pts: np.ndarray, 
                          ransac_threshold: float = 2.0, max_iters: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        N = len(src_pts)
        if N < 4:
            return np.eye(3), np.zeros(N, dtype=bool)

        best_H = None
        max_inliers = 0
        best_mask = np.zeros(N, dtype=bool)

        src_h = np.hstack((src_pts, np.ones((N, 1))))

        for _ in range(max_iters):
            indices = np.random.choice(N, 4, replace=False)
            
            try:
                H_sample = self.compute_dlt(src_pts[indices], dst_pts[indices])
            except Exception:
                continue
                
            projected = (H_sample @ src_h.T).T
            
            denom = projected[:, 2]
            valid_idx = np.abs(denom) > 1e-10
            
            proj_x = projected[valid_idx, 0] / denom[valid_idx]
            proj_y = projected[valid_idx, 1] / denom[valid_idx]
            
            dx = proj_x - dst_pts[valid_idx, 0]
            dy = proj_y - dst_pts[valid_idx, 1]
            errors = np.sqrt(dx**2 + dy**2)
            
            current_mask = np.zeros(N, dtype=bool)
            current_mask[valid_idx] = errors < ransac_threshold
            num_inliers = np.sum(current_mask)

            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_H = H_sample
                best_mask = current_mask
                
                if num_inliers > 0.95 * N:
                    break
        
        if best_H is not None and max_inliers > 4:
            refined_H = self.compute_dlt(src_pts[best_mask], dst_pts[best_mask])
            return refined_H, best_mask
            
        if best_H is None:
            return np.eye(3), best_mask
            
        return best_H, best_mask

    def _create_soft_weights(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        mask1 = (gray1 > 0).astype(np.uint8)
        mask2 = (gray2 > 0).astype(np.uint8)
        
        dist1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 5).astype(np.float32)
        dist2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 5).astype(np.float32)
        
        sum_dist = dist1 + dist2
        sum_dist[sum_dist == 0] = 1.0
        
        weight1 = dist1 / sum_dist
        weight2 = dist2 / sum_dist
        
        return weight1, weight2

    def _blend_core(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        h, w = img1.shape[:2]
        
        w1, w2 = self._create_soft_weights(img1, img2)
        
        w1 = w1[:, :, np.newaxis]
        w2 = w2[:, :, np.newaxis]
        
        img1_float = img1.astype(np.float32)
        img2_float = img2.astype(np.float32)
        
        result = (img1_float * w1 + img2_float * w2)
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def stitch(self, img1: np.ndarray, img2: np.ndarray, 
               pts1: np.ndarray, pts2: np.ndarray,
               use_ransac: bool = True,
               ransac_threshold: float = 3.0,
               use_blending: bool = True,
               use_cylindrical: bool = False,
               focal_length: float = None) -> np.ndarray:
        
        if len(pts1) < 4:
            raise ValueError("Can it nhat 4 cap diem khop.")
        
        if use_cylindrical:
            img1 = self.cylindrical_projection(img1, focal_length)
            img2 = self.cylindrical_projection(img2, focal_length)
        
        if use_ransac:
            H, _ = self._ransac_homography(pts1, pts2, ransac_threshold)
        else:
            H = self.compute_dlt(pts1, pts2)
        
        if np.abs(np.linalg.det(H)) < 1e-8:
            return img1
        
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        corners_1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners_1_warped = cv2.perspectiveTransform(corners_1, H)
        
        corners_2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        
        all_corners = np.concatenate((corners_1_warped, corners_2), axis=0)
        
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        translation = [-xmin, -ymin]
        H_translation = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])
        
        output_h, output_w = ymax - ymin, xmax - xmin
        
        img1_warped = cv2.warpPerspective(img1, H_translation @ H, (output_w, output_h))
        
        img2_canvas = np.zeros_like(img1_warped)
        y_start, x_start = translation[1], translation[0]
        img2_canvas[y_start:y_start+h2, x_start:x_start+w2] = img2
        
        if use_blending:
            result = self._blend_core(img1_warped, img2_canvas)
        else:
            result = img1_warped.copy()
            result[y_start:y_start+h2, x_start:x_start+w2] = img2
        
        return result