import cv2
import numpy as np
from pathlib import Path

def crop_black_borders(img, threshold=10):
    if img is None or img.size == 0:
        return img
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img
    
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y+h, x:x+w]
    
    return cropped


def crop_largest_rectangle(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 1, cv2.THRESH_BINARY)
    
    rows, cols = mask.shape
    
    heights = np.zeros((rows, cols), dtype=np.int32)
    
    for r in range(rows):
        if r == 0:
            heights[r] = mask[r]
        else:
            heights[r] = (heights[r-1] + 1) * mask[r]
    
    max_area = 0
    best_rect = (0, 0, 0, 0)
    
    for r in range(rows):
        row_heights = heights[r]
        
        stack = [-1]
        current_heights = np.append(row_heights, 0)
        
        for i, h in enumerate(current_heights):
            while stack[-1] != -1 and current_heights[stack[-1]] >= h:
                h_idx = stack.pop()
                height = current_heights[h_idx]
                width = i - stack[-1] - 1
                area = height * width
                
                if area > max_area:
                    max_area = area
                    best_rect = (stack[-1] + 1, r - height + 1, width, height)
                    
            stack.append(i)
    
    x, y, w, h = best_rect
    
    if max_area > 0:
        result = img[y:y+h, x:x+w]
        return result
    else:
        return img


def images_stitching(image_paths, matcher, stitcher, output_dir, 
                     crop_borders=True,
                     use_cylindrical=False,
                     focal_length=800,
                     ransac_threshold=2.0,
                     max_keypoints=8192):
    n_images = len(image_paths)
    if n_images < 2:
        print("Can it nhat 2 anh.")
        return None, False
    
    result = cv2.imread(image_paths[0])
    if result is None:
        print(f"Loi: Khong doc duoc anh {image_paths[0]}.")
        return None, False
    
    success = True
    for i in range(1, n_images):
        img_name_1 = Path(image_paths[i-1]).name if i == 1 else "result"
        img_name_2 = Path(image_paths[i]).name
        
        next_img = cv2.imread(image_paths[i])
        if next_img is None:
            print(f"Loi: Khong doc duoc anh.")
            success = False
            break
        
        size_1 = f"{result.shape[1]}x{result.shape[0]}"
        size_2 = f"{next_img.shape[1]}x{next_img.shape[0]}"
        print(f"\nGhep {img_name_1} ({size_1}) <-> {img_name_2} ({size_2})")
        
        temp_result_path = str(Path(output_dir) / f'temp_result_{i-1}.jpg')
        cv2.imwrite(temp_result_path, result)
        
        try:
            pts1, pts2 = matcher.match_images(temp_result_path, image_paths[i])
            
            if len(pts1) < 10:
                print(f"Canh bao: Chi co {len(pts1)} diem.")
                success = False
                break
            
            result = stitcher.stitch(result, next_img, pts1, pts2, 
                                    use_ransac=True, 
                                    use_blending=True,
                                    use_cylindrical=use_cylindrical,
                                    focal_length=focal_length,
                                    ransac_threshold=ransac_threshold)
            
            if result.shape[0] > 4096 or result.shape[1] > 4096:
                before_size = f"{result.shape[1]}x{result.shape[0]}"
                max_dim = max(result.shape[0], result.shape[1])
                scale = 4096 / max_dim
                new_h = int(result.shape[0] * scale)
                new_w = int(result.shape[1] * scale)
                result = cv2.resize(result, (new_w, new_h))
                after_size = f"{result.shape[1]}x{result.shape[0]}"
                print(f"Resize: {before_size} -> {after_size}")
            
            if crop_borders:
                result = crop_black_borders(result, threshold=10)
            
            intermediate_path = Path(output_dir) / f'step_{i}_stitched.jpg'
            cv2.imwrite(str(intermediate_path), result)
            
        except Exception as e:
            print(f"Loi: {str(e)}")
            success = False
            break
    
    result_uncropped = None
    if result is not None and success:
        result_uncropped = result.copy()
    
    if result is not None and crop_borders and success:
        print(f"\nTim hinh chu nhat noi tiep lon nhat")
        before_size = f"{result.shape[1]}x{result.shape[0]}"
        result = crop_largest_rectangle(result)
        after_size = f"{result.shape[1]}x{result.shape[0]}"
        print(f"Crop: {before_size} -> {after_size}")
    
    return result, result_uncropped, success
