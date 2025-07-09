"""
Multi-Camera Tracking ReID Feature Extractor
Integrates Fast-Reid with ByteTrack for appearance-based person re-identification
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import os
import sys

# Add fast-reid to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../fast-reid-mct'))

try:
    from fastreid.config import get_cfg
    from fastreid.engine import DefaultPredictor
    from fastreid.utils.logger import setup_logger
    FASTREID_AVAILABLE = True
except ImportError:
    FASTREID_AVAILABLE = False
    print("Warning: Fast-Reid not available. MCT will use basic features.")


class ReidExtractor:
    """
    Fast-Reid feature extractor for Multi-Camera Tracking.
    Extracts appearance features from person detections for cross-camera association.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 model_path: Optional[str] = None,
                 device: str = "cuda",
                 input_size: Tuple[int, int] = (256, 128)):
        """
        Initialize the ReID feature extractor.
        
        Args:
            config_path: Path to Fast-Reid config file
            model_path: Path to Fast-Reid model weights
            device: Device to run inference on
            input_size: Input size for ReID model (height, width)
        """
        self.device = device
        self.input_size = input_size
        self.enabled = FASTREID_AVAILABLE
        
        if not self.enabled:
            print("Fast-Reid not available. Using dummy features.")
            return
            
        # Setup default config if not provided
        if config_path is None:
            config_path = self._get_default_config()
        if model_path is None:
            model_path = self._get_default_model()
            
        self.config_path = config_path
        self.model_path = model_path
        
        # Initialize Fast-Reid
        self._setup_fastreid()
        
    def _get_default_config(self) -> str:
        """Get default Fast-Reid configuration."""
        # Use Market1501 bagtricks config as default
        config_dir = os.path.join(os.path.dirname(__file__), '../../fast-reid-mct/configs/Market1501')
        config_file = os.path.join(config_dir, 'bagtricks_R50-ibn.yml')
        
        if os.path.exists(config_file):
            return config_file
        else:
            # Fallback to any available config
            configs = [
                'bagtricks_R50.yml',
                'bagtricks_R101-ibn.yml',
                'sbs_R50-ibn.yml'
            ]
            for config in configs:
                config_path = os.path.join(config_dir, config)
                if os.path.exists(config_path):
                    return config_path
                    
        raise FileNotFoundError("No Fast-Reid config found")
        
    def _get_default_model(self) -> str:
        """Get default Fast-Reid model path."""
        # Common model paths
        model_paths = [
            "pretrained/fastreid_market1501_bagtricks_R50-ibn.pth",
            "pretrained/fastreid_market1501_bagtricks_R50.pth",
            "fast-reid-mct/pretrained/market1501_bagtricks_R50-ibn.pth",
            "fast-reid-mct/pretrained/market1501_bagtricks_R50.pth"
        ]
        
        for model_path in model_paths:
            full_path = os.path.join(os.path.dirname(__file__), '../../', model_path)
            if os.path.exists(full_path):
                return full_path
                
        print("Warning: No Fast-Reid model found. Will use config defaults.")
        return ""
        
    def _setup_fastreid(self):
        """Initialize Fast-Reid model."""
        try:
            setup_logger(name="fastreid")
            
            # Load config
            cfg = get_cfg()
            cfg.merge_from_file(self.config_path)
            
            # Set device
            cfg.MODEL.DEVICE = self.device
            
            # Set input size
            cfg.INPUT.SIZE_TEST = self.input_size
            
            # Set model weights if available
            if self.model_path and os.path.exists(self.model_path):
                cfg.MODEL.WEIGHTS = self.model_path
            
            cfg.freeze()
            
            # Create predictor
            self.predictor = DefaultPredictor(cfg)
            self.cfg = cfg
            
            print(f"Fast-Reid initialized successfully with config: {self.config_path}")
            
        except Exception as e:
            print(f"Error initializing Fast-Reid: {e}")
            self.enabled = False
            
    def extract_features(self, 
                        image: np.ndarray, 
                        bboxes: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """
        Extract ReID features from person detections.
        
        Args:
            image: Input image (H, W, C) in BGR format
            bboxes: Person bounding boxes in format [x1, y1, x2, y2] or [x, y, w, h]
            
        Returns:
            features: Normalized feature vectors (N, D) where N is number of boxes
        """
        if not self.enabled:
            return self._extract_dummy_features(len(bboxes))
            
        if len(bboxes) == 0:
            return np.empty((0, 2048), dtype=np.float32)  # Default feature dimension
            
        # Convert bboxes to numpy array
        bboxes = np.array(bboxes)
        
        # Extract person crops
        person_crops = self._extract_person_crops(image, bboxes)
        
        # Extract features from crops
        features = []
        for crop in person_crops:
            if crop is not None:
                feature = self._extract_single_feature(crop)
                features.append(feature)
            else:
                # Use zero feature for invalid crops
                features.append(np.zeros(2048, dtype=np.float32))
                
        return np.array(features)
        
    def _extract_person_crops(self, 
                             image: np.ndarray, 
                             bboxes: np.ndarray) -> List[Optional[np.ndarray]]:
        """Extract person crops from image using bounding boxes."""
        crops = []
        h, w = image.shape[:2]
        
        for bbox in bboxes:
            # Convert bbox format if needed
            if len(bbox) == 4:
                # Check if format is [x, y, w, h] or [x1, y1, x2, y2]
                if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
                    # Likely [x, y, w, h] format
                    x1, y1, w_box, h_box = bbox
                    x2, y2 = x1 + w_box, y1 + h_box
                else:
                    # Likely [x1, y1, x2, y2] format
                    x1, y1, x2, y2 = bbox
            else:
                crops.append(None)
                continue
                
            # Clip to image boundaries
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            
            # Extract crop
            if x2 > x1 and y2 > y1:
                crop = image[y1:y2, x1:x2]
                crops.append(crop)
            else:
                crops.append(None)
                
        return crops
        
    def _extract_single_feature(self, crop: np.ndarray) -> np.ndarray:
        """Extract feature from single person crop."""
        try:
            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            crop_resized = cv2.resize(crop_rgb, self.input_size[::-1], 
                                    interpolation=cv2.INTER_CUBIC)
            
            # Convert to tensor
            crop_tensor = torch.as_tensor(crop_resized.astype("float32").transpose(2, 0, 1))[None]
            
            # Move to device
            crop_tensor = crop_tensor.to(self.device)
            
            # Extract feature
            with torch.no_grad():
                features = self.predictor(crop_tensor)
                
            # Normalize features
            features = F.normalize(features, dim=1)
            
            return features.cpu().numpy()[0]
            
        except Exception as e:
            print(f"Error extracting feature: {e}")
            return np.zeros(2048, dtype=np.float32)
            
    def _extract_dummy_features(self, num_boxes: int) -> np.ndarray:
        """Extract dummy features when Fast-Reid is not available."""
        # Return random normalized features for testing
        features = np.random.randn(num_boxes, 512).astype(np.float32)
        # Normalize features
        features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        return features
        
    def compute_similarity(self, 
                          features1: np.ndarray, 
                          features2: np.ndarray,
                          metric: str = "cosine") -> np.ndarray:
        """
        Compute similarity between two sets of features.
        
        Args:
            features1: First set of features (N, D)
            features2: Second set of features (M, D)
            metric: Similarity metric ("cosine", "euclidean")
            
        Returns:
            similarity_matrix: Similarity scores (N, M)
        """
        if metric == "cosine":
            # Cosine similarity
            similarity = np.dot(features1, features2.T)
        elif metric == "euclidean":
            # Euclidean distance (converted to similarity)
            from scipy.spatial.distance import cdist
            distances = cdist(features1, features2, metric='euclidean')
            similarity = 1.0 / (1.0 + distances)
        else:
            raise ValueError(f"Unknown metric: {metric}")
            
        return similarity
        
    def is_available(self) -> bool:
        """Check if Fast-Reid is available and properly initialized."""
        return self.enabled
        
    def get_feature_dim(self) -> int:
        """Get the dimension of extracted features."""
        if self.enabled:
            return 2048  # Default Fast-Reid feature dimension
        else:
            return 512   # Dummy feature dimension