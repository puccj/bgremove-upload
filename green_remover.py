import cv2
import numpy as np
import os
import argparse
from config import *

def advanced_chroma_key(
        frame, 
        bg, 
        green_lower=GREEN_LOWER, 
        green_upper=GREEN_UPPER, 
        feather_radius=FEATHER_RADIUS, 
        spill_suppress=SPILL_SUPPRESS,
        skin_warmth=SKIN_WARMTH):
    """
    Perform advanced chroma keying with soft masking and color decontamination.
    Returns the composited frame with the background replaced.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Create a soft mask for green areas
    mask = cv2.inRange(hsv, green_lower, green_upper).astype(np.float32) / 255.0
    mask = cv2.GaussianBlur(mask, (feather_radius, feather_radius), 0)
    mask = np.clip(mask, 0, 1)
    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    # Color decontamination (reduce green spill) on keyed (green) areas
    frame_no_spill = frame.copy().astype(np.float32)
    green = frame_no_spill[..., 1]
    red_blue = (frame_no_spill[..., 0] + frame_no_spill[..., 2]) / 2
    spill_mask = (green > red_blue) & (mask > 0.05)
    frame_no_spill[..., 1][spill_mask] = red_blue[spill_mask] * (1 - spill_suppress) + frame_no_spill[..., 1][spill_mask] * spill_suppress
    
    # Additional despill on the foreground (areas kept), helps faces
    if DESPILL_FG > 0:
        fg_mask = (1.0 - mask)
        fg_spill = (green > red_blue) & (fg_mask > 0.3)
        # Move green toward the average of red/blue
        frame_no_spill[..., 1][fg_spill] = green[fg_spill] - DESPILL_FG * (green[fg_spill] - red_blue[fg_spill])

    # Optional skin-tone warming on the foreground
    if skin_warmth > 0:
        frame_warm = apply_skin_warmth(frame_no_spill, mask, skin_warmth)
    else:
        frame_warm = frame_no_spill

    # Alpha blend with background
    mask_3c = np.stack([mask]*3, axis=-1)
    result = frame_warm * (1 - mask_3c) + bg.astype(np.float32) * mask_3c
    return result.astype(np.uint8)

def apply_skin_warmth(frame_float_bgr: np.ndarray, mask_green: np.ndarray, strength: float) -> np.ndarray:
    """
    Warm up skin tones in the foreground (non-green areas).
    
    Parameters
    ----------
    frame_float_bgr : np.ndarray
        Input image in BGR format as float32.
    mask_green : np.ndarray
        Soft mask of green areas (0-1).
    strength : float
        Strength of the warmth effect (0-1).
    
    Returns
    -------
    np.ndarray
        The image with warmed skin tones.
    """

    # Foreground mask
    fg = 1.0 - mask_green
    
    # Detect skin in YCrCb
    bgr_u8 = np.clip(frame_float_bgr, 0, 255).astype(np.uint8)
    ycrcb = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    skin_mask = (Cr >= 135) & (Cr <= 185) & (Cb >= 85) & (Cb <= 135)
    
    # Combine with foreground and feather
    skin_mask = skin_mask.astype(np.float32) * fg
    skin_mask = cv2.GaussianBlur(skin_mask, (9, 9), 0)
    skin_mask = np.clip(skin_mask, 0, 1)

    if skin_mask.max() < 0.01:
        return frame_float_bgr

    # Convert to LAB for more natural warming (increase a,b toward red/yellow)
    lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)
    A_warm = A + (8.0 * strength)
    B_warm = B + (16.0 * strength)
    A_warm = np.clip(A_warm, 0, 255)
    B_warm = np.clip(B_warm, 0, 255)
    lab_warm = cv2.merge([L, A_warm, B_warm]).astype(np.uint8)
    bgr_warm = cv2.cvtColor(lab_warm, cv2.COLOR_LAB2BGR).astype(np.float32)

    # Blend only on skin areas of the foreground
    skin_mask_3c = np.stack([skin_mask]*3, axis=-1)
    out = frame_float_bgr * (1 - skin_mask_3c) + bgr_warm * skin_mask_3c
    return out

def load_image(image_path):
    """Load an image from the specified path."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image from {image_path}.")
    return img

def load_background(bg_path, image_shape):
    """Load and resize background image to match input image size."""
    bg = cv2.imread(bg_path)
    if bg is None:
        raise FileNotFoundError(f"Cannot load background image from {bg_path}.")
    return cv2.resize(bg, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser(description="Green Screen Remover for a single image")
    parser.add_argument("--input", type=str, default=INPUT_IMAGE, help="Path to input image with green screen")
    parser.add_argument("--bg", type=str, default=BACKGROUND_IMAGE, help="Path to background image")
    parser.add_argument("--output", type=str, default=None, help="Path to output image (optional)")
    parser.add_argument("--test", action="store_true", help="Perform green removal with multiple parameters to later check the best result")

    args = parser.parse_args()
    input_image_path = args.input
    background_image_path = args.bg
    output_path = args.output
    if output_path is None:
        base_name = os.path.basename(input_image_path)
        name, ext = os.path.splitext(base_name)
        output_path = f"{name}_no_green{ext}"
    test_param = args.test

    # Load input image and background with the same size
    image = load_image(input_image_path)
    bg_img = load_background(background_image_path, image.shape)

    if test_param:
        os.makedirs("test_params", exist_ok=True)

        for preset in ["broad", "default", "narrowed"]:
            if preset == "broad":
                g_low = np.array([40, 100, 50]) # HSV lower bound for green
                g_up = np.array([80, 255, 255]) # HSV upper bound for green
            elif preset == "default":
                g_low = np.array([50, 120, 60])
                g_up = np.array([75, 255, 255])
            elif preset == "narrowed":
                g_low = np.array([56, 140, 70])
                g_up = np.array([68, 255, 255])
            for feature_radius in range(7, 21, 2):
                for spill_suppress in np.arange(0.5, 1.0, 0.1):
                    for warm_strength in np.arange(0.0, 1.0, 0.25):
                        file_name = f"Preset {preset}, Feather {feature_radius}, Spill {spill_suppress:.1f}, Warmth {warm_strength:.1f}.jpg"
                        result = advanced_chroma_key(image, bg_img, g_low, g_up, feature_radius, spill_suppress, warm_strength)
                        cv2.imwrite(f"test_params/{file_name}", result)
                        print(f"File {file_name} saved.")
        print("Test images saved in 'test_params' folder. Please review them to find the best parameters.")
        return
    
    # Apply advanced chroma keying
    result = advanced_chroma_key(image, bg_img, GREEN_LOWER, GREEN_UPPER, FEATHER_RADIUS, SPILL_SUPPRESS, SKIN_WARMTH)

    # Save the result
    cv2.imwrite(output_path, result)
    print(f"Output saved to {output_path}")

    # Display the result
    cv2.imshow("Original", image)
    cv2.imshow("Result", result)
    print("Press any key to close the preview windows...")
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()