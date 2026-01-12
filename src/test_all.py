import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lightglue import viz2d
from matcher import FeatureMatcher
from stitcher import ImageStitcher
from datetime import datetime

def test_all(image_path0, image_path1, output_dir='../outputs/test'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = str(Path(output_dir) / timestamp)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    matcher = FeatureMatcher(max_num_keypoints=2048)
    stitcher = ImageStitcher()
    
    print("Doc anh...")
    img1_cv = cv2.imread(image_path0)
    img2_cv = cv2.imread(image_path1)
    print(f"Anh 1: {img1_cv.shape}")
    print(f"Anh 2: {img2_cv.shape}")
    
    print("\nTim diem khop...")
    image0, _ = matcher.load_image(image_path0)
    image1, _ = matcher.load_image(image_path1)
    feats0, feats1, matches01 = matcher.match_pair(image0, image1)
    
    kpts0 = feats0['keypoints']
    kpts1 = feats1['keypoints']
    matches = matches01['matches']
    m_kpts0 = kpts0[matches[..., 0]]
    m_kpts1 = kpts1[matches[..., 1]]
    
    print(f"Tim thay {len(matches)} cap diem khop")
    print("\nVisualize matches...")

    fig = plt.figure(figsize=(16, 8))
    axes = viz2d.plot_images([image0.cpu(), image1.cpu()])
    viz2d.plot_matches(m_kpts0.cpu(), m_kpts1.cpu(), color='lime', lw=0.5)
    viz2d.add_text(0, f'{len(matches)} matches', fs=15)
    
    viz_path = str(Path(output_dir) / 'matches_visualization.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"Da luu: {viz_path}")
    plt.close()
    print("\nLay toa do diem khop...")
    pts1, pts2 = matcher.match_images(image_path0, image_path1)
    print("\nTinh Homography va ghep anh...")

    try:
        result = stitcher.stitch(img1_cv, img2_cv, pts1, pts2, use_ransac=True)
        print(f"Kich thuoc anh ghep: {result.shape}")
        
        stitch_path = str(Path(output_dir) / 'stitched_result.jpg')
        cv2.imwrite(stitch_path, result)
        print(f"Da luu: {stitch_path}")
        
        print("\nHoan thanh.")
    except Exception as e:
        print(f"\nLoi: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Test stitching')
    parser.add_argument('--images', nargs=2, help='2 anh can test')
    parser.add_argument('--folder', type=str, help='Thu muc chua anh')
    parser.add_argument('--output_dir', type=str, default='../outputs/test', help='Thu muc luu ket qua')
    args = parser.parse_args()
    
    if args.folder:
        folder = Path(args.folder)
        if not folder.exists():
            print(f"Loi: Khong tim thay folder {args.folder}")
            return
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_paths = []
        for ext in extensions:
            image_paths.extend(sorted(folder.glob(ext)))
        
        if len(image_paths) < 2:
            print(f"Loi: Chi tim thay {len(image_paths)} anh trong folder")
            return
        
        print(f"Tim thay {len(image_paths)} anh, dung 2 anh dau:")
        print(f"  1. {image_paths[0].name}")
        print(f"  2. {image_paths[1].name}\n")
        
        test_all(str(image_paths[0]), str(image_paths[1]), args.output_dir)
    elif args.images:
        test_all(args.images[0], args.images[1], args.output_dir)
    else:
        print("Loi: Can chi dinh --images hoac --folder")
        parser.print_help()

if __name__ == '__main__':
    main()
