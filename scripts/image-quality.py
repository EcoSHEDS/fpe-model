import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_image(path):
    """Load image from path, returning None if failed."""
    try:
        img = cv2.imread(str(path))
        if img is None:
            logger.warning(f"Failed to load image: {path}")
        return img
    except Exception as e:
        logger.error(f"Error loading image {path}: {e}")
        return None

def image_quality(image):
    """Compute various image quality metrics.
    
    Args:
        image: Input image array in BGR format
        
    Returns:
        dict: Dictionary containing computed metrics
    """
    metrics = {}
    
    # Convert to different color spaces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Check if grayscale (by comparing channels)
    b, g, r = cv2.split(image)
    is_grayscale = np.allclose(r, g, rtol=1e-1, atol=1e-1) and np.allclose(g, b, rtol=1e-1, atol=1e-1)
    metrics['is_grayscale'] = float(is_grayscale)
    
    # Mean RGB values
    metrics['mean_r'] = float(np.mean(r))
    metrics['mean_g'] = float(np.mean(g))
    metrics['mean_b'] = float(np.mean(b))
    
    # Lightness metrics (from LAB color space)
    l_channel = lab[:,:,0]
    metrics['mean_lightness'] = float(np.mean(l_channel))
    metrics['std_lightness'] = float(np.std(l_channel))
    
    # Dark and light fractions
    metrics['fraction_dark'] = float(np.mean(l_channel < 64))
    metrics['fraction_light'] = float(np.mean(l_channel > 192))
    
    # Sharpness using Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    metrics['variance_laplacian'] = float(laplacian.var())
    
    # Haze score using dark channel prior
    kernel_size = 15
    dark_channel = np.min(image, axis=2)
    dark_channel = cv2.erode(dark_channel, np.ones((kernel_size, kernel_size)))
    metrics['haze_score'] = float(np.mean(dark_channel))
    
    # Saturation from HSV
    metrics['mean_saturation'] = float(np.mean(hsv[:,:,1]))
    
    # Contrast metrics
    metrics['rms_contrast'] = float(np.std(gray) / np.mean(gray) if np.mean(gray) > 0 else 0)
    metrics['michelson_contrast'] = float((np.max(gray) - np.min(gray)) / (np.max(gray) + np.min(gray) + 1e-6))
    
    # Edge density
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    metrics['edge_density'] = float(np.mean(edge_magnitude))
    
    # Noise estimation
    median_filtered = cv2.medianBlur(gray, 3)
    noise_estimate = np.mean(np.abs(gray - median_filtered))
    metrics['noise_level'] = float(noise_estimate)
    
    # Glare detection
    metrics['glare_fraction'] = float(np.mean((gray > 240) & (hsv[:,:,1] < 30)))
    
    # Local contrast variation
    local_std = cv2.blur(gray, (15,15))
    metrics['local_contrast_variation'] = float(np.std(local_std))
    
    # Motion blur detection using FFT
    rows, cols = gray.shape
    crow, ccol = rows//2, cols//2
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    metrics['frequency_energy'] = float(np.mean(magnitude_spectrum[crow-30:crow+30, ccol-30:ccol+30]))
    
    # Entropy
    histogram = cv2.calcHist([gray],[0],None,[256],[0,256])
    histogram_normalized = histogram / histogram.sum()
    non_zero_hist = histogram_normalized[histogram_normalized > 0]
    metrics['entropy'] = float(-np.sum(non_zero_hist * np.log2(non_zero_hist)))
    
    return metrics

def process_single_image(img_path, img_dir):
    """Process a single image and return its metrics.
    
    Args:
        img_path: Path to image file
        img_dir: Root directory containing images
        
    Returns:
        dict: Image metrics or None if processing failed
    """
    full_path = img_dir / img_path
    img = load_image(full_path)
    if img is not None:
        try:
            return image_quality(img)
        except Exception as e:
            logger.error(f"Error processing {full_path}: {e}")
    return {k: None for k in image_quality(np.zeros((100,100,3), dtype=np.uint8))}

def process_images(input_file: Path, img_dir: Path, n: int = None, output_file: Path = None, n_cores: int = None):
    """Process images in parallel and append quality metrics to input CSV."""
    # Read input CSV
    df = pd.read_csv(input_file)
    if 'filename' not in df.columns:
        raise ValueError("Input CSV must contain 'filename' column")
    
    # Limit number of images if specified
    if n is not None:
        df = df.head(n)
    
    # Determine number of cores to use
    if n_cores is None:
        n_cores = cpu_count() - 1
    n_cores = min(n_cores, cpu_count() - 1)
    
    logger.info(f"Processing {len(df)} images from {input_file} using {n_cores} cores")
    
    # Process images in parallel
    with Pool(n_cores) as pool:
        process_func = partial(process_single_image, img_dir=img_dir)
        
        # Process images and show progress
        metrics_list = []
        start_time = time.time()
        
        for i, metrics in enumerate(pool.imap(process_func, df['filename'])):
            metrics_list.append(metrics)
            
            if (i + 1) % 100 == 0:
                elapsed_time = time.time() - start_time
                images_per_sec = (i + 1) / elapsed_time
                remaining_images = len(df) - (i + 1)
                eta_seconds = remaining_images / images_per_sec
                eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
                
                logger.info(
                    f"Processed {i + 1}/{len(df)} images "
                    f"({images_per_sec:.1f} img/s, "
                    f"ETA: {eta_str})"
                )
    
    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    
    # Combine with original DataFrame
    result_df = pd.concat([df, metrics_df], axis=1)
    
    # Save results
    if output_file is None:
        output_file = input_file.with_stem(input_file.stem + '_metrics')
    
    result_df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")
    
    return result_df

def parse_args():
    parser = argparse.ArgumentParser(description='Compute image quality metrics for a dataset')
    parser.add_argument('--input-file', type=Path, required=True,
                      help='Input CSV file containing image filenames')
    parser.add_argument('--img-dir', type=Path, required=True,
                      help='Root directory containing images')
    parser.add_argument('--n', type=int, default=None,
                      help='Number of images to process (default: all)')
    parser.add_argument('--output-file', type=Path, default=None,
                      help='Output CSV file (default: input_file with _metrics suffix)')
    parser.add_argument('--n-cores', type=int, default=None,
                      help='Number of CPU cores to use (default: all available except 1)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    process_images(
        args.input_file, 
        args.img_dir, 
        args.n, 
        args.output_file,
        args.n_cores
    )

