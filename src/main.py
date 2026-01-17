import argparse
import cv2
from pathlib import Path
from datetime import datetime
from matcher import FeatureMatcher
from stitcher import ImageStitcher
from utils import crop_black_borders, images_stitching


def main():
    parser = argparse.ArgumentParser(description='image stitching')
    parser.add_argument('--images', nargs='+', help='danh sach anh')
    parser.add_argument('--folder', type=str, help='thu muc chua anh')
    parser.add_argument('--output_dir', type=str, default='../outputs/panorama', help='thu muc output')
    parser.add_argument('--max_keypoints', type=int, default=8192, help='so keypoint toi da')
    parser.add_argument('--no_crop', action='store_true', help='khong cat vien den')
    parser.add_argument('--cylindrical', action='store_true', help='dung phep chieu tru')
    parser.add_argument('--focal_length', type=float, help='do dai tieu cu')
    args = parser.parse_args()
    
    if args.folder:
        folder = Path(args.folder)
        if not folder.exists():
            print(f"Loi: Khong tim thay folder {args.folder}")
            return
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_paths = []
        for ext in extensions:
            image_paths.extend(folder.glob(ext))
        
        image_paths = sorted(list(set(image_paths)))
        
        if len(image_paths) < 2:
            print(f"Loi: Chi tim thay {len(image_paths)} anh trong folder")
            return
        
        image_paths = [str(p) for p in image_paths]
    elif args.images:
        image_paths = args.images
    else:
        print("Loi: Can chi dinh --images hoac --folder")
        return
    
    if len(image_paths) < 2:
        print("Loi: Can it nhat 2 anh.")
        return
    
    if len(image_paths) > 5:
        print(f"Canh bao: Dang ghep {len(image_paths)} anh - nen dung tu 5 anh tro xuong de tranh drift")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSo anh: {len(image_paths)}, bao gom:")
    for i, img in enumerate(image_paths, 1):
        print(f"{i}. {Path(img).name}")
        
    max_kp = args.max_keypoints if hasattr(args, 'max_keypoints') and args.max_keypoints else 8192
    focal = args.focal_length if hasattr(args, 'focal_length') and args.focal_length else 800
    use_cyl = args.cylindrical if hasattr(args, 'cylindrical') else False
    
    matcher = FeatureMatcher(max_num_keypoints=max_kp)
    stitcher = ImageStitcher(focal_length=focal)
    
    no_crop = args.no_crop if hasattr(args, 'no_crop') else False
    
    result, result_uncropped, success = images_stitching(
        image_paths,
        matcher,
        stitcher,
        output_dir,
        crop_borders=not no_crop,
        use_cylindrical=use_cyl,
        focal_length=focal,
        ransac_threshold=2.0,
        max_keypoints=max_kp
    )
    
    if result is not None:
        final_path_cropped = output_dir / 'panorama_cropped.jpg'
        cv2.imwrite(str(final_path_cropped), result)
        
        if result_uncropped is not None:
            final_path_uncropped = output_dir / 'panorama_uncropped.jpg'
            cv2.imwrite(str(final_path_uncropped), result_uncropped)
        
        if success:
            print(f"\nHoan thanh.")
            print(f"Kich thuoc panorama (da crop): {result.shape}")
            print(f"Luu tai:")
            print(f"- Anh da crop: {final_path_cropped}")
            if result_uncropped is not None:
                print(f"- Anh chua crop: {final_path_uncropped}")
        else:
            print(f"\nGhep mot phan thanh cong (co loi xay ra).")
            print(f"Kich thuoc panorama hien tai: {result.shape}")
            print(f"Luu tai: {final_path_cropped}")
    else:
        print("\nLoi: Khong ghep duoc panorama.")


if __name__ == '__main__':
    main()
