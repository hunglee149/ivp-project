import argparse
import cv2
import numpy as np
from pathlib import Path
from matcher import FeatureMatcher
from stitcher import ImageStitcher

def test_stitching(image_path0, image_path1, output_path=None):
    print("\nKhoi tao Matcher & Stitcher...")
    matcher = FeatureMatcher(max_num_keypoints=2048)
    stitcher = ImageStitcher()
    
    print("Doc anh...")
    img1 = cv2.imread(image_path0)
    img2 = cv2.imread(image_path1)
    
    if img1 is None or img2 is None:
        print("Loi: Khong doc duoc anh!")
        return
    
    print(f"Anh 1: {img1.shape}")
    print(f"Anh 2: {img2.shape}")
    
    print("\nTim diem khop...")
    pts1, pts2 = matcher.match_images(image_path0, image_path1)
    print(f"Tim thay {len(pts1)} cap diem")
    
    if len(pts1) < 4:
        print("Loi: Can it nhat 4 cap diem!")
        return
    
    print("\nTinh Homography va ghep anh...")
    try:
        result = stitcher.stitch(img1, img2, pts1, pts2, use_ransac=True)
        print(f"Kich thuoc anh ghep: {result.shape}")
        
        if output_path:
            cv2.imwrite(output_path, result)
            print(f"Da luu: {output_path}")
        
        print("\nHien thi ket qua...")
        result_resized = cv2.resize(result, None, fx=0.5, fy=0.5)
        cv2.imshow('Stitched Result (Press any key to close)', result_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("\nHoan thanh!")
        
    except Exception as e:
        print(f"\nLoi: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Test Image Stitching')
    parser.add_argument('--images', nargs=2, required=True, help='2 anh can ghep')
    parser.add_argument('--output', type=str, default='stitched_result.jpg', help='Duong dan luu anh')
    args = parser.parse_args()
    
    test_stitching(args.images[0], args.images[1], args.output)

if __name__ == '__main__':
    main()
