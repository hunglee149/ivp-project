import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from lightglue import viz2d
from matcher import FeatureMatcher

def visualize_match_pair(matcher, image_path0, image_path1):
    print(f"\nTim diem tuong dong giua: {Path(image_path0).name} <-> {Path(image_path1).name}")
    
    image0, _ = matcher.load_image(image_path0)
    image1, _ = matcher.load_image(image_path1)
    
    feats0, feats1, matches01 = matcher.match_pair(image0, image1)
    
    kpts0 = feats0['keypoints']
    kpts1 = feats1['keypoints']
    matches = matches01['matches']
    
    m_kpts0 = kpts0[matches[..., 0]]
    m_kpts1 = kpts1[matches[..., 1]]
    
    print(f"Tim thay {len(matches)} cap diem tuong dong.")

    axes = viz2d.plot_images([image0.cpu(), image1.cpu()])
    
    viz2d.plot_matches(m_kpts0.cpu(), m_kpts1.cpu(), color='lime', lw=0.5)
    
    viz2d.add_text(0, f'{len(matches)} matches', fs=15)
    
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--max_keypoints', type=int, default=2048)
    args = parser.parse_args()
    
    if len(args.images) < 2:
        print("Can it nhat 2 anh de chay!")
        return

    matcher = FeatureMatcher(max_num_keypoints=args.max_keypoints)
    visualize_match_pair(matcher, args.images[0], args.images[1])

if __name__ == '__main__':
    main()
