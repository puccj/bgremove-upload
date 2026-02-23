import numpy as np
from typing import Set

IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

####################################
#   Green  removal configuration   #
####################################

# Here three presets. Feel free to adjust or create your own.
preset = "default"  # Options: "broad", "default", "narrowed"
if preset == "broad":
    GREEN_LOWER = np.array([40, 100, 50])   # HSV lower bound for green
    GREEN_UPPER = np.array([80, 255, 255])  # HSV upper bound for green
elif preset == "default":
    GREEN_LOWER = np.array([50, 120, 60])
    GREEN_UPPER = np.array([75, 255, 255])
elif preset == "narrowed":
    GREEN_LOWER = np.array([56, 140, 70])
    GREEN_UPPER = np.array([68, 255, 255])

FEATHER_RADIUS = 13     # Edge feathering (must be odd)
SPILL_SUPPRESS = 0.7    # Spill suppression strength
DESPILL_FG = 0.5        # Additional despill on foreground (0-1)
SKIN_WARMTH = 0.25      # Warmth strength applied to skin (0-1)

INPUT_IMAGE = "input.jpg"           # Path to input image with green screen
BACKGROUND_IMAGE = "background.jpg" # Path to background image
OUTPUT_PATH = "output.jpg"           # Folder to save output images


############################
#  Preview configuration   #
############################

PREVIEW_MAX_WIDTH = 1280
PREVIEW_MAX_HEIGHT = 720
UPDATE_INTERVAL_PREVIEW = 0.1   # Seconds between preview loop updates
MIN_AGE_PREVIEW = 0.0			# Minimum age of file to be considered for preview (to avoid processing files still being written)

HEARTBEAT_SECONDS = 5.0	# Interval for logging heartbeat when no new images are found
						# Use 0 to disable heartbeat logging
