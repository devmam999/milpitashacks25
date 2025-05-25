import logging
from typing import Dict, List, Tuple
from PIL import Image, UnidentifiedImageError
import numpy as np
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FER detector with proper error handling
detector = None
try:
    from fer import FER
    detector = FER(mtcnn=True)
    logger.info("FER detector initialized successfully")
except ImportError as e:
    logger.error(f"FER library not installed: {e}")
    raise ImportError("Please install the FER library: pip install fer")
except Exception as e:
    logger.error(f"Failed to initialize FER detector: {e}")
    raise RuntimeError(f"FER initialization failed: {str(e)}")

# Emotion severity scale
distress_levels = {
    "happy": {"color": "green", "scale": (7, 8), "description": "Positive"},
    "neutral": {"color": "yellow", "scale": (5, 6), "description": "Neutral"},
    "surprise": {"color": "yellow", "scale": (5, 6), "description": "Neutral"},
    "sad": {"color": "orange", "scale": (3, 4), "description": "Concerning"},
    "fear": {"color": "orange", "scale": (3, 4), "description": "Concerning"},
    "disgust": {"color": "orange", "scale": (3, 4), "description": "Concerning"},
    "angry": {"color": "red", "scale": (1, 2), "description": "Needs Support"},
}

def load_image_from_bytes(contents: bytes) -> np.ndarray:
    """Load and decode image from raw bytes."""
    if not contents:
        raise ValueError("Empty file contents")
    
    if len(contents) > 10 * 1024 * 1024:
        raise ValueError("File size exceeds 10MB limit")
    
    try:
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Validate image dimensions
        if image.size[0] < 100 or image.size[1] < 100:
            raise ValueError("Image too small; minimum size is 100x100 pixels")
        
        # Resize if too large
        max_size = (640, 480)
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image = image.resize(max_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_np = np.array(image, dtype=np.uint8)
        
        # Validate the numpy array
        if (
            image_np is None or
            not isinstance(image_np, np.ndarray) or
            image_np.dtype != np.uint8 or
            image_np.ndim != 3 or
            image_np.shape[2] != 3
        ):
            raise ValueError("Image conversion failed. Please provide a valid RGB image.")
            
        logger.info(f"Image loaded successfully: shape {image_np.shape}")
        return image_np
        
    except UnidentifiedImageError:
        raise ValueError("File could not be identified as an image.")
    except Exception as e:
        logger.error(f"Failed to decode image: {str(e)}")
        raise ValueError(f"Invalid image format: {str(e)}")

def detect_faces_and_emotions(image: np.ndarray) -> List[Dict]:
    """Detect faces and emotions in the image."""
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image data. Please provide a valid image.")
    
    if detector is None:
        raise RuntimeError("FER detector not initialized")
    
    try:
        logger.info(f"Processing image with shape: {image.shape}")
        faces = detector.detect_emotions(image)
        
        if faces is None:
            logger.warning("FER returned None - no faces detected or processing failed")
            return []
            
        if not isinstance(faces, list):
            logger.error(f"FER returned unexpected type: {type(faces)}")
            raise RuntimeError("FER failed to process the image properly.")
            
        logger.info(f"Detected {len(faces)} faces")
        
    except Exception as e:
        logger.error(f"Face detection failed: {str(e)}")
        raise RuntimeError(f"Failed to analyze face: {str(e)}. Ensure the image is clear and well-lit.")
    
    if not faces:
        logger.info("No faces detected in the image")
        return []
    
    results = []
    for i, face_data in enumerate(faces):
        try:
            # Validate face data structure
            if not isinstance(face_data, dict):
                logger.warning(f"Face {i}: Invalid face data structure")
                continue
                
            box = face_data.get("box")
            if not box or not isinstance(box, (list, tuple)) or len(box) != 4:
                logger.warning(f"Face {i}: Invalid bounding box data")
                continue
            
            x, y, w, h = box
            emotions = face_data.get("emotions", {})
            
            if not emotions or not isinstance(emotions, dict):
                logger.warning(f"Face {i}: No emotions detected, using neutral")
                emotion, score = "neutral", 0.5
                severity_info = distress_levels["neutral"]
            else:
                # Find the dominant emotion
                emotion = max(emotions, key=emotions.get)
                score = emotions[emotion]
                
                # Special case for very happy
                if emotion == "happy" and score >= 0.9:
                    severity_info = {"color": "blue", "scale": (9, 10), "description": "Very Positive"}
                else:
                    severity_info = distress_levels.get(
                        emotion, 
                        {"color": "yellow", "scale": (5, 6), "description": "Neutral"}
                    )
            
            # Ensure all values are properly typed
            result = {
                "face_location": {
                    "top": int(y), 
                    "right": int(x + w), 
                    "bottom": int(y + h), 
                    "left": int(x)
                },
                "emotion": str(emotion),
                "score": float(score),
                "severity_color": str(severity_info["color"]),
                "severity_scale": tuple(int(val) for val in severity_info["scale"]),
                "severity_description": str(severity_info["description"]),
            }
            
            results.append(result)
            logger.info(f"Face {i}: {emotion} ({score:.3f}) at {box}")
            
        except Exception as e:
            logger.error(f"Error processing face {i}: {str(e)}")
            continue
    
    return results

def analyze_emotion(image: np.ndarray) -> Tuple[str, float, Dict]:
    """Analyze emotions in the image and return the primary emotion."""
    try:
        results = detect_faces_and_emotions(image)
        
        if not results:
            logger.info("No faces detected, returning default response")
            return "no face", 0.0, {"color": "gray", "scale": (0, 0), "description": "No face detected"}
        
        # Use the first (presumably most prominent) face
        first_face = results[0]
        
        # Ensure clean data types for JSON serialization
        distress_clean = {
            "color": str(first_face["severity_color"]),
            "scale": tuple(int(x) for x in first_face["severity_scale"]),
            "description": str(first_face["severity_description"])
        }
        
        emotion = str(first_face["emotion"])
        score = float(first_face["score"])
        
        logger.info(f"Analysis complete: {emotion} ({score:.3f})")
        return emotion, score, distress_clean
        
    except Exception as e:
        logger.error(f"Error in analyze_emotion: {str(e)}")
        raise RuntimeError(f"Emotion analysis failed: {str(e)}")

# Optional: Helper function for batch processing
def analyze_multiple_faces(image: np.ndarray) -> List[Dict]:
    """Analyze all faces in the image and return detailed results."""
    try:
        results = detect_faces_and_emotions(image)
        
        processed_results = []
        for face in results:
            processed_face = {
                "emotion": face["emotion"],
                "confidence": face["score"],
                "location": face["face_location"],
                "distress": {
                    "color": face["severity_color"],
                    "scale": face["severity_scale"],
                    "description": face["severity_description"]
                }
            }
            processed_results.append(processed_face)
        
        return processed_results
        
    except Exception as e:
        logger.error(f"Error in analyze_multiple_faces: {str(e)}")
        raise RuntimeError(f"Multiple face analysis failed: {str(e)}")