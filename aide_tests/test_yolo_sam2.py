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
import time

# Add the project root to the path to import YOLOv10
sys.path.append('/scratch/sy2366/Project/multirobot_infi_llm')
# from detect.ultralytics import YOLOv10
from ultralytics import YOLO
# from segment_anything import SamPredictor
# from MobileSAM.setup_mobile_sam import setup_model
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import torch

# Prompting SAM with detected boxes
def segment(sam_predictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
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
    Checks if CUDA is available and moves model to CUDA if possible.
    """
    model = YOLO("yolov10m.pt")
    if torch.cuda.is_available():
        model.to('cuda')
        print("YOLO model loaded on CUDA.")
    else:
        print("YOLO model loaded on CPU.")
    return model

def load_sam_model(model_path="/scratch/sy2366/Project/multirobot_infi_llm/aide_tests/mobile_sam.pt"):

    checkpoint = "./sam2_hiera_large.pt"
    model_cfg = "./sam2_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))



    return predictor

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
    start_time = time.time()
    results = model(image, conf=0.5)
    detection_time = time.time() - start_time
    print(f"   Detection time: {detection_time:.3f} seconds")
    
    
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
            pdb.set_trace()
            for i in range(len(boxes)):
                detections.append({
                    'bbox': boxes[i],
                    'confidence': confidences[i],
                    'class_id': class_ids[i],
                    'class_name': class_names.get(class_ids[i], f"class_{class_ids[i]}")
                })
    
    return image,image_rgb, detections,results,detection_time



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

def process_single_image(model, sam_predictor, image_path, output_dir, confidence_threshold=0.2, no_display=True):
    """
    Process a single image with YOLO detection and SAM segmentation.
    """
    # Extract image name for output files
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print(f"{'='*60}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
    
    # Perform detection
    print("1. Performing object detection...")
    image_bgr, image, detections, results, detection_time = perform_detection(
        model, image_path, confidence_threshold=confidence_threshold
    )
    
    if image is None:
        print("Failed to process image.")
        return None
    
    print(f"2. Detection completed. Found {len(detections)} objects:")
    for i, det in enumerate(detections):
        print(f"   {i+1}. {det['class_name']} (confidence: {det['confidence']:.3f})")
    
    # Plot detection results
    print("3. Plotting detection results...")
    if no_display:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    
    detection_output_path = os.path.join(output_dir, f"{image_name}_detection_results.png")
    start_time = time.time()
    plot_detections(image, detections, save_path=detection_output_path, show_plot=not no_display)
    plotting_time = time.time() - start_time
    print(f"   Plotting time: {plotting_time:.3f} seconds")
    
    # Perform SAM segmentation
    print("4. Performing SAM Segmentation...")
    if len(detections) == 0:
        print("   No objects detected, skipping segmentation.")
        return {
            'image_name': image_name,
            'detection_time': detection_time,
            'plotting_time': plotting_time,
            'segmentation_time': 0,
            'num_detections': 0
        }
    
    # Convert detections to masks
    bbox = results[0].boxes.xyxy
    start_time = time.time()
    masks = segment(
        sam_predictor=sam_predictor,
        image=image,
        xyxy=bbox
    )
    segmentation_time = time.time() - start_time
    print(f"   Segmentation time: {segmentation_time:.3f} seconds")
    
    # Convert masks from float to boolean
    if masks is not None and len(masks) > 0:
        masks = np.array(masks) > 0.5
    
    # Create supervision detections object
    xyxy = np.array(bbox.cpu().numpy())
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

    # Annotate image with masks
    mask_annotator = sv.MaskAnnotator()
    annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=sv_detections)
    
    # Save segmentation results
    segmentation_output_path = os.path.join(output_dir, f"{image_name}_segmentation_results.png")
    cv2.imwrite(segmentation_output_path, annotated_image)
    print(f"   Segmentation results saved to: {segmentation_output_path}")
    
    return {
        'image_name': image_name,
        'detection_time': detection_time,
        'plotting_time': plotting_time,
        'segmentation_time': segmentation_time,
        'num_detections': len(detections),
        'detection_output': detection_output_path,
        'segmentation_output': segmentation_output_path
    }

def main():
    """
    Main function to run YOLO detection and SAM segmentation on multiple images.
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='YOLO Detection and SAM Segmentation Test')
    parser.add_argument('--images', '-i', nargs='+',
                       default=["/scratch/sy2366/Project/multirobot_infi_llm/aide_tests/obs1.png",
                               "/scratch/sy2366/Project/multirobot_infi_llm/aide_tests/obs2.png"],
                       help='Paths to input images')
    parser.add_argument('--output-dir', '-o',
                       default="/scratch/sy2366/Project/multirobot_infi_llm/aide_tests/",
                       help='Directory to save output images')
    parser.add_argument('--model', '-m', default=None,
                       help='Path to YOLO model weights')
    parser.add_argument('--confidence', '-c', type=float, default=0.2,
                       help='Confidence threshold for detection')
    parser.add_argument('--no-display', action='store_true', default=True,
                       help='Do not display the plot (save only)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("YOLO Detection and SAM Segmentation Test")
    print("=" * 80)
    print(f"Input images: {args.images}")
    print(f"Output directory: {args.output_dir}")
    print(f"Confidence threshold: {args.confidence}")
    print("=" * 80)
    
    # Load YOLO model
    print("1. Loading YOLO model...")
    start_time = time.time()
    model = load_yolo_model(args.model)
    yolo_load_time = time.time() - start_time
    print(f"   YOLO model loading time: {yolo_load_time:.3f} seconds")
    if model is None:
        print("Failed to load YOLO model. Exiting.")
        return
    
    # Load SAM model
    print("2. Loading SAM model...")
    start_time = time.time()
    sam_predictor = load_sam_model()
    sam_load_time = time.time() - start_time
    print(f"   SAM model loading time: {sam_load_time:.3f} seconds")
    if sam_predictor is None:
        print("Failed to load SAM model. Exiting.")
        return
    
    # Process each image
    results = []
    total_processing_start = time.time()
    
    for i, image_path in enumerate(args.images):
        print(f"\n{'='*80}")
        print(f"Processing image {i+1}/{len(args.images)}: {os.path.basename(image_path)}")
        print(f"{'='*80}")
        
        result = process_single_image(
            model=model,
            sam_predictor=sam_predictor,
            image_path=image_path,
            output_dir=args.output_dir,
            confidence_threshold=args.confidence,
            no_display=args.no_display
        )
        
        if result:
            results.append(result)
    
    total_processing_time = time.time() - total_processing_start
    
    # Print summary
    print(f"\n{'='*80}")
    print("PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total images processed: {len(results)}")
    print(f"Total processing time: {total_processing_time:.3f} seconds")
    print(f"Average time per image: {total_processing_time/len(results):.3f} seconds")
    print(f"\nModel loading times:")
    print(f"  YOLO model: {yolo_load_time:.3f} seconds")
    print(f"  SAM model:  {sam_load_time:.3f} seconds")
    
    print(f"\nPer-image results:")
    for result in results:
        print(f"  {result['image_name']}:")
        print(f"    Detections: {result['num_detections']}")
        print(f"    Detection time: {result['detection_time']:.3f}s")
        print(f"    Segmentation time: {result['segmentation_time']:.3f}s")
        print(f"    Plotting time: {result['plotting_time']:.3f}s")
        if 'detection_output' in result:
            print(f"    Detection output: {result['detection_output']}")
        if 'segmentation_output' in result:
            print(f"    Segmentation output: {result['segmentation_output']}")
    
    print(f"\n{'='*80}")
    print("All processing completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
