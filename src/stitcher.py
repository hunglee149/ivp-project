import numpy as np
import cv2
from typing import Tuple, Optional

class ImageStitcher:
    def __init__(self):
        pass

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
            raise ValueError("Cần ít nhất 4 cặp điểm để tính homography")
        
        if len(src_pts) != len(dst_pts):
            raise ValueError("Số điểm nguồn và đích phải bằng nhau")
        
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
                raise ValueError("Homography suy biến (H[2,2] ≈ 0) - có thể do điểm thẳng hàng hoặc nhiễu lớn")
            
            H_norm = H_norm / H_norm[2, 2]
            H = np.linalg.inv(T_dst) @ H_norm @ T_src
            H = H / H[2, 2]
            return H
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"Không thể tính SVD: {str(e)}")

    def _ransac_homography(self, src_pts: np.ndarray, dst_pts: np.ndarray, 
                          ransac_threshold: float = 3.0, max_iters: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
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

        print(f"RANSAC: {max_inliers}/{N} inliers ({max_inliers/N*100:.1f}%)")
        
        if best_H is not None and max_inliers > 4:
            refined_H = self.compute_dlt(src_pts[best_mask], dst_pts[best_mask])
            return refined_H, best_mask
            
        if best_H is None:
            return np.eye(3), best_mask
            
        return best_H, best_mask

    def _linear_blend(self, img1: np.ndarray, img2: np.ndarray, max_size=3072) -> np.ndarray:
        h, w = img1.shape[:2]
        
        if max(h, w) > max_size:
            try:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                
                img1_small = cv2.resize(img1, (new_w, new_h))
                img2_small = cv2.resize(img2, (new_w, new_h))
                
                blended_small = self._blend_core(img1_small, img2_small)
                
                blended = cv2.resize(blended_small, (w, h))
                return blended
            except (MemoryError, cv2.error, np.core._exceptions._ArrayMemoryError):
                mask2_any = (img2.sum(axis=2) > 0)
                result = img1.copy()
                result[mask2_any] = img2[mask2_any]
                return result
        else:
            return self._blend_core(img1, img2)
    
    def _blend_core(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        try:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
            
            mask1 = (gray1 > 0).astype(np.uint8)
            mask2 = (gray2 > 0).astype(np.uint8)
            overlap = mask1 & mask2
            
            if not overlap.any():
                return np.where(mask1[:,:,None] > 0, img1, img2)
            
            dist1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 5)
            dist2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 5)
            
            weight1 = dist1 / (dist1 + dist2 + 1e-10)
            weight2 = 1 - weight1
            
            result = np.zeros_like(img1, dtype=np.float32)
            for c in range(3):
                result[:,:,c] = img1[:,:,c] * weight1 + img2[:,:,c] * weight2
            
            return result.astype(np.uint8)
        except (MemoryError, np.core._exceptions._ArrayMemoryError):
            mask2_any = (img2.sum(axis=2) > 0)
            result = img1.copy()
            result[mask2_any] = img2[mask2_any]
            return result

    def stitch(self, img1: np.ndarray, img2: np.ndarray, 
               pts1: np.ndarray, pts2: np.ndarray,
               use_ransac: bool = True,
               ransac_threshold: float = 3.0,
               use_blending: bool = True) -> np.ndarray:
        
        if len(pts1) < 4:
            raise ValueError("Cần ít nhất 4 cặp điểm khớp")
        
        if use_ransac:
            H, _ = self._ransac_homography(pts1, pts2, ransac_threshold)
        else:
            H = self.compute_dlt(pts1, pts2)
        
        det = np.linalg.det(H[:2, :2])
        if abs(det) < 1e-6:
            raise ValueError(f"Homography không hợp lệ (det={det:.2e})")
        
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        corners = np.array([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]], dtype=np.float32).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H)
        
        all_corners = np.concatenate((warped_corners, 
                                      np.array([[0, 0], [0, h2-1], [w2-1, h2-1], [w2-1, 0]], dtype=np.float32).reshape(-1, 1, 2)))
        
        xmin, ymin = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        xmax, ymax = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        width = xmax - xmin
        height = ymax - ymin
        
        shift = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float64)
        H_final = shift @ H
        
        result = cv2.warpPerspective(img1, H_final, (width, height))
        
        y_start = -ymin
        x_start = -xmin
        y_end = y_start + h2
        x_end = x_start + w2
        
        result_y1 = max(0, y_start)
        result_x1 = max(0, x_start)
        result_y2 = min(height, y_end)
        result_x2 = min(width, x_end)
        
        img2_y1 = result_y1 - y_start
        img2_x1 = result_x1 - x_start
        img2_y2 = img2_y1 + (result_y2 - result_y1)
        img2_x2 = img2_x1 + (result_x2 - result_x1)
        
        if use_blending:
            canvas = np.zeros_like(result)
            canvas[result_y1:result_y2, result_x1:result_x2] = img2[img2_y1:img2_y2, img2_x1:img2_x2]
            result = self._linear_blend(result, canvas)
        else:
            result[result_y1:result_y2, result_x1:result_x2] = img2[img2_y1:img2_y2, img2_x1:img2_x2]
        
        return result