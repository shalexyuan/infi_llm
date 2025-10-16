#!/usr/bin/env python3
"""
YOLO Detection Test Script
Reads obs1.png, performs YOLO detection, and plots bounding boxes with categories.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import sys
import pdb
import supervision as sv

# Add the project root to the path to import YOLOv10
sys.path.append('/scratch/sy2366/Project/multirobot_infi_llm')
from detect.ultralytics import YOLOv10
from segment_anything import SamPredictor
from MobileSAM.setup_mobile_sam import setup_model
import torch

  # Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

def load_yolo_model(model_path="jameslahm/yolov10m"):
    """
    Load YOLO model. If no model_path is provided, use a default pretrained model.
    """
    return  YOLOv10.from_pretrained('jameslahm/yolov10m')

def load_sam_model(model_path="/scratch/sy2366/Project/multirobot_infi_llm/aide_tests/mobile_sam.pt"):
    # Building MobileSAM predictor
    MOBILE_SAM_CHECKPOINT_PATH = model_path
    checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)
    mobile_sam = setup_model()
    mobile_sam.load_state_dict(checkpoint, strict=True)
    mobile_sam.to(device="cuda")

    sam_predictor = SamPredictor(mobile_sam)
    return sam_predictor

def perform_detection(model, image_path, confidence_threshold=0.2):
    """
    Perform YOLO detection on the image.
    """

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"Image shape: {image_rgb.shape}")
    
    # Perform detection
    results = model(image, conf=0.5)
    
    # Extract detection results
    detections = []
    if len(results) > 0:
        result = results[0]
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            # Get class names
            if hasattr(result, 'names'):
                class_names = result.names
            else:
                class_names = {i: f"class_{i}" for i in range(80)}  # Default COCO classes
            
            for i in range(len(boxes)):
                detections.append({
                    'bbox': boxes[i],
                    'confidence': confidences[i],
                    'class_id': class_ids[i],
                    'class_name': class_names.get(class_ids[i], f"class_{class_ids[i]}")
                })
    
    return image,image_rgb, detections



def plot_detections(image, detections, save_path=None, show_plot=True):
    """
    Plot the image with bounding boxes and labels.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    # Define colors for different classes
    if len(detections) > 0:
        colors = plt.cm.Set3(np.linspace(0, 1, len(detections)))
    else:
        colors = []
    
    for i, detection in enumerate(detections):
        bbox = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name']
        print(f"Detection: {detection}, bbox: {bbox}")
        
        # Extract coordinates
        x1, y1, x2, y2 = bbox
        
        # Create rectangle
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=colors[i], facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        label = f"{class_name}: {confidence:.2f}"
        ax.text(x1, y1-5, label, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.7),
                fontsize=10, color='black')
    
    ax.set_title(f"YOLO Detection Results ({len(detections)} objects detected)", fontsize=14)
    ax.axis('off')
    
    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig

def main():
    """
    Main function to run YOLO detection test.
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='YOLO Detection Test')
    parser.add_argument('--image', '-i', 
                       default="/scratch/sy2366/Project/multirobot_infi_llm/aide_tests/obs1.png",
                       help='Path to input image')
    parser.add_argument('--output', '-o',
                       default="/scratch/sy2366/Project/multirobot_infi_llm/aide_tests/yolo_detection_results.png",
                       help='Path to save output image')
    parser.add_argument('--model', '-m', default=None,
                       help='Path to YOLO model weights')
    parser.add_argument('--confidence', '-c', type=float, default=0.2,
                       help='Confidence threshold for detection')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display the plot (save only)')
    
    args = parser.parse_args()
    
    # Paths
    image_path = args.image
    output_path = args.output
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    print("=" * 60)
    print("YOLO Detection Test")
    print("=" * 60)
    print(f"Input image: {image_path}")
    print(f"Output image: {output_path}")
    print(f"Confidence threshold: {args.confidence}")
    print("=" * 60)
    
    # Load YOLO model
    print("1. Loading YOLO model...")
    model = load_yolo_model(args.model)
    if model is None:
        print("Failed to load YOLO model. Exiting.")
        return
    
    
    # Perform detection
    print("2. Performing object detection...")
    image_bgr,image, detections = perform_detection(model, image_path, confidence_threshold=args.confidence)
    
    if image is None:
        print("Failed to process image. Exiting.")
        return
    
    print(f"3. Detection completed. Found {len(detections)} objects:")
    for i, det in enumerate(detections):
        print(f"   {i+1}. {det['class_name']} (confidence: {det['confidence']:.3f})")
    
    # Plot results
    print("4. Plotting results...")
    if args.no_display:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    

    plot_detections(image, detections, save_path=output_path, show_plot=not args.no_display)
    
    print("=" * 60)
    print("YOLO Detection Test Completed!")
    print(f"Results saved to: {output_path}")
    print("=" * 60)


    print("5. Loading SAM model...")
    sam_predictor = load_sam_model()
    if sam_predictor is None:
        print("Failed to load SAM model. Exiting.")
        return

    print("6. Performing SAM Segmentation...")
    # convert detections to masks
    bbox=[d['bbox'] for d in detections]
    masks = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        xyxy=bbox
    )

    pdb.set_trace()


    xyxy = np.array(bbox)
    confidence = np.array([d['confidence'] for d in detections])
    class_id = np.array([d['class_id'] for d in detections])
    mask = np.array(masks) if len(masks) > 0 else None

    # Create sv.Detections object
    sv_detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        mask=mask
    )

    mask_annotator = sv.MaskAnnotator()
    annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=sv_detections)
    
    cv2.imwrite("/scratch/sy2366/Project/multirobot_infi_llm/aide_tests/sam_results.png", annotated_image)
    print(f"Results saved to: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
