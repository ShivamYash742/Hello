import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt

class FeatureBasedAlignment:
    """Feature-based registration for optical-thermal alignment."""
    
    def __init__(self, 
                 detector_type: str = 'ORB',
                 max_features: int = 5000,
                 match_ratio: float = 0.75):
        self.detector_type = detector_type
        self.max_features = max_features
        self.match_ratio = match_ratio
        
        # Initialize detector
        if detector_type == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=max_features)
        elif detector_type == 'AKAZE':
            self.detector = cv2.AKAZE_create()
        elif detector_type == 'SIFT':
            self.detector = cv2.SIFT_create(nfeatures=max_features)
        else:
            raise ValueError(f"Unsupported detector: {detector_type}")
            
        # Matcher
        if detector_type in ['ORB', 'AKAZE']:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    def detect_and_match(self, 
                        img1: np.ndarray, 
                        img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect keypoints and match descriptors."""
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        else:
            gray1 = img1
            
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            gray2 = img2
        
        # Detect keypoints and descriptors
        kp1, desc1 = self.detector.detectAndCompute(gray1, None)
        kp2, desc2 = self.detector.detectAndCompute(gray2, None)
        
        if desc1 is None or desc2 is None:
            return np.array([]), np.array([])
        
        # Match descriptors
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:
            return np.array([]), np.array([])
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        return pts1, pts2
    
    def estimate_transform(self, 
                          pts1: np.ndarray, 
                          pts2: np.ndarray,
                          method: str = 'homography') -> Tuple[np.ndarray, np.ndarray]:
        """Estimate geometric transformation using RANSAC."""
        if len(pts1) < 4:
            return None, None
        
        if method == 'homography':
            transform, mask = cv2.findHomography(
                pts1, pts2, 
                cv2.RANSAC, 
                ransacReprojThreshold=5.0,
                confidence=0.99
            )
        elif method == 'affine':
            transform, mask = cv2.estimateAffinePartial2D(
                pts1, pts2,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0,
                confidence=0.99
            )
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return transform, mask
    
    def align_images(self, 
                    reference: np.ndarray,
                    moving: np.ndarray,
                    method: str = 'homography') -> Dict:
        """Align moving image to reference."""
        # Detect and match features
        pts_ref, pts_mov = self.detect_and_match(reference, moving)
        
        if len(pts_ref) == 0:
            return {
                'aligned': moving,
                'transform': None,
                'inliers': 0,
                'total_matches': 0,
                'success': False
            }
        
        # Estimate transformation
        transform, mask = self.estimate_transform(pts_ref, pts_mov, method)
        
        if transform is None:
            return {
                'aligned': moving,
                'transform': None,
                'inliers': 0,
                'total_matches': len(pts_ref),
                'success': False
            }
        
        # Apply transformation
        h, w = reference.shape[:2]
        if method == 'homography':
            aligned = cv2.warpPerspective(moving, transform, (w, h))
        else:
            aligned = cv2.warpAffine(moving, transform, (w, h))
        
        inliers = np.sum(mask) if mask is not None else 0
        
        return {
            'aligned': aligned,
            'transform': transform,
            'inliers': inliers,
            'total_matches': len(pts_ref),
            'success': True
        }
    
    def compute_residuals(self, 
                         pts1: np.ndarray, 
                         pts2: np.ndarray, 
                         transform: np.ndarray) -> np.ndarray:
        """Compute reprojection residuals."""
        if transform.shape[0] == 3:  # Homography
            pts1_h = cv2.convertPointsToHomogeneous(pts1)
            pts2_proj = cv2.perspectiveTransform(pts1, transform)
        else:  # Affine
            pts2_proj = cv2.transform(pts1, transform)
        
        residuals = np.linalg.norm(pts2 - pts2_proj, axis=2).flatten()
        return residuals
    
    def save_alignment_plot(self, 
                           reference: np.ndarray,
                           moving: np.ndarray, 
                           aligned: np.ndarray,
                           save_path: str):
        """Save alignment visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(reference, cmap='gray')
        axes[0].set_title('Reference')
        axes[0].axis('off')
        
        axes[1].imshow(moving, cmap='gray')
        axes[1].set_title('Moving')
        axes[1].axis('off')
        
        axes[2].imshow(aligned, cmap='gray')
        axes[2].set_title('Aligned')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()