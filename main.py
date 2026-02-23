"""
This script continuously monitors a specified folder for new images, applies chroma keying to remove green backgrounds, 
and uploads the processed images to Google Photos. It also provides a real-time preview of the latest image with the background removed.
"""


import argparse
import multiprocessing as mp
import os
import random
import time
import traceback
from pathlib import Path
from typing import Dict, List, Set, Tuple

import requests
import cv2
import numpy as np

from green_remover import (
	advanced_chroma_key,
	load_background,
	load_image,
)

from upload_utils import (
	get_session,
    load_manifest, 
    save_manifest, 
    hash_file, 
    upload_data, 
    create_media_item
)

from config import *

def find_new_images(folder: Path,
					manifest: Dict[str, str],
					min_age_sec: float = 2.0,
					hash_cache: Dict[Path, Tuple[int, int, str]] | None = None,
					) -> List[Tuple[Path, str]]:
	"""Return (image_path, file_hash) for images not present in manifest."""
	now = time.time()
	manifest_hashes = set(manifest)
	candidates: List[Tuple[Path, str]] = []

	if hash_cache is None:
		hash_cache = {}

	scanned_paths: Set[Path] = set()
	for path in folder.rglob("*"):
		if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
			continue
		scanned_paths.add(path)
		try:
			stat = path.stat()
		except FileNotFoundError:
			continue

		if (now - stat.st_mtime) < min_age_sec:
			continue

		mtime_ns = stat.st_mtime_ns
		size = stat.st_size
		cached = hash_cache.get(path)
		if cached and cached[0] == mtime_ns and cached[1] == size:
			file_hash = cached[2]
		else:
			file_hash = hash_file(path)
			hash_cache[path] = (mtime_ns, size, file_hash)

		if file_hash not in manifest_hashes:
			candidates.append((path, file_hash))

	for cached_path in list(hash_cache):
		if cached_path not in scanned_paths:
			hash_cache.pop(cached_path, None)

	return candidates

def get_latest_image(folder: Path, min_age_sec: float = 0.0) -> Path | None:
	"""Return the newest image in folder, or None if no stable image exists."""
	now = time.time()
	latest_path: Path | None = None
	latest_mtime = -1.0
	for path in folder.rglob("*"):
		if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
			continue
		try:
			stat = path.stat()
		except FileNotFoundError:
			continue
		if (now - stat.st_mtime) < min_age_sec:
			continue
		if stat.st_mtime > latest_mtime:
			latest_mtime = stat.st_mtime
			latest_path = path
	return latest_path

def resize(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
	"""Resize image to fit max dimensions while preserving aspect ratio."""
	if image is None:
		return image
	h, w = image.shape[:2]
	if w <= max_width and h <= max_height:
		return image
	scale = min(max_width / w, max_height / h)
	return cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

# def list_top_level_images(folder: Path) -> List[Path]:
# 	"""List supported images in top-level folder only."""
# 	images: List[Path] = []
# 	for path in folder.iterdir():
# 		if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
# 			images.append(path)
# 	return images


def remove_green(img_path: Path, image: np.ndarray, bg_folder: str) -> Tuple[bytes, str, np.ndarray]:
	"""Remove green screen from the image and return (encoded_bytes, output_name, result_array).
	The background is randomly chosen from the specified folder."""
	
	if image is None:
		raise ValueError(f"Cannot load image: {img_path}")
	
	# Randomly select a background image from the folder
	bg_folder_path = Path(bg_folder)
	if not bg_folder_path.exists() or not bg_folder_path.is_dir():
		raise ValueError(f"Background folder not found: {bg_folder}")
	
	bg_images = [f for f in bg_folder_path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
	if not bg_images:
		raise ValueError(f"No background images found in: {bg_folder}")
	
	bg_path = random.choice(bg_images)
	# print(f"  Using background: {bg_path.name}")
	bg_img = load_background(str(bg_path), image.shape)
	result = advanced_chroma_key(image, bg_img)

	# Choose encoding based on original extension; fallback to jpg
	ext_map = {".png": ".png", ".jpg": ".jpg", ".jpeg": ".jpg", ".webp": ".webp", ".bmp": ".bmp", ".tiff": ".tiff"}
	enc_ext = ext_map.get(img_path.suffix.lower(), ".jpg")
	ok, buf = cv2.imencode(enc_ext, result)
	if not ok:
		raise RuntimeError("Failed to encode processed image")
	data = buf.tobytes()
	out_name = f"{img_path.stem}_no_green{enc_ext}"
	return data, out_name, result

def display_image(window_name: str, image: np.ndarray, old_image: np.ndarray | None = None, transition_time: float = 1.0) -> None:
	"""Display an image in a window, optionally with a fade transition."""
	
	# Add a simple fade transition if old_image is provided
	if old_image is not None and transition_time > 0:
		old_disp = old_image
		if old_image.shape[1] > 1920 or old_image.shape[0] > 1080:
			scale = min(1920 / old_image.shape[1], 1080 / old_image.shape[0])
			old_disp = cv2.resize(old_image, (0, 0), fx=scale, fy=scale)
		
		for alpha in np.linspace(0, 1, num=int(transition_time * 30)):
			blend = cv2.addWeighted(image, alpha, old_disp, 1 - alpha, 0)
			cv2.imshow(window_name, blend)
			cv2.waitKey(30)
	else:
		cv2.imshow(window_name, image)
		cv2.waitKey(1)  # Needed to update the window

def preview_loop(
		folder: Path, 
		bg_folder: str, 
		window_name: str = "Preview",
		transition_time: float = 0.0,
		interval: float = UPDATE_INTERVAL_PREVIEW,
		min_age: float = MIN_AGE_PREVIEW
	) -> None:
	"""Continuously preview the latest image in low quality."""
	
	last_preview_key: Tuple[str, int, int] | None = None
	last_heartbeat = 0.0
	
	# Display the background as an initial placeholder
	try:
		bg_example = [f for f in Path(bg_folder).iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS][0]
		bg_img = load_background(str(bg_example), (1920, 1080))
		display_image(window_name, bg_img)
	except (IndexError, ValueError):
		print(f"[preview] no background images found in: {bg_folder}")

	while True:
		should_break = False
		try:
			latest = get_latest_image(folder, min_age_sec=min_age)
			if latest is None:
				if HEARTBEAT_SECONDS > 0:
					now = time.time()
					if (now - last_heartbeat) >= HEARTBEAT_SECONDS:
						print(f"[preview] Waiting | images in {folder}={len(list(folder.glob('*')))}")
						last_heartbeat = now
				time.sleep(interval)
				continue

			try:
				stat = latest.stat()
			except FileNotFoundError:
				time.sleep(interval)
				continue

			# Avoid re-processing the same file (based on path, mtime, and size)
			preview_key = (str(latest), stat.st_mtime_ns, stat.st_size)
			if preview_key == last_preview_key:
				time.sleep(interval)
				continue

			original = load_image(str(latest))
			if original is None:
				time.sleep(interval)
				continue
			low_quality = resize(original, PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT)
			_, _, preview_result = remove_green(latest, low_quality, bg_folder)
			display_image(window_name, preview_result, original, transition_time)
			print(f"[preview] showing {latest.name}")
			last_preview_key = preview_key
		except Exception as exc:  # pylint: disable=broad-except
			print(f"[preview] error: {exc}")
		finally:
			if cv2.waitKey(1) & 0xFF == ord("q"):
				should_break = True
			time.sleep(interval)
		if should_break:
			break

	cv2.destroyAllWindows()
	print("[preview] stopped")


def upload_loop(
		folder: Path,
		credentials: str,
		token: str,
		manifest_path: Path,
		bg_folder: str = "backgrounds",
		album_id: str | None = None,
		output_folder: Path | None = None,
		interval: float = 5.0,
		min_age: float = 2.0
	) -> None:
	"""
	Continuously process and upload high-quality images.

	Parameters
	----------
	folder: Path
		Folder to monitor for new images.
	credentials: str
		Path to OAuth client credentials JSON.
	token: str
		Path to stored user token.
	manifest_path: Path
		Path to manifest file.
	bg_folder: str
		Name of background images folder (relative to folder).
	album_id: str | None
		ID of album to add media items to, or None to upload to library only.
	output_folder: Path | None
		Folder to save output images in, or None for no output saving.
	interval: float
		Time in seconds between scans for new images.
	min_age: float
		Minimum age in seconds of an image before it is processed.
	"""

	manifest = load_manifest(manifest_path)
	session = get_session(credentials, token)
	hash_cache: Dict[Path, Tuple[int, int, str]] = {}
	last_heartbeat = 0.0

	if output_folder is not None:
		output_folder.mkdir(parents=True, exist_ok=True)

	while True:
		new_images = find_new_images(folder, manifest, min_age_sec=min_age, hash_cache=hash_cache)
		if not new_images:
			now = time.time()
			if (now - last_heartbeat) >= HEARTBEAT_SECONDS:
				print(
					f"[upload] scan heartbeat | images_in_folder={len(list(folder.glob('*')))} "
					f"manifest_entries={len(manifest)} new_images=0"
				)
				last_heartbeat = now
			time.sleep(interval)
			continue

		print(f"[upload] found {len(new_images)} new image(s)")
		for img_path, file_hash in new_images:
			try:
				original_image = load_image(str(img_path))
				if original_image is None:
					raise ValueError(f"Cannot load image: {img_path}")
				data_bytes, out_name, result_arr = remove_green(img_path, original_image, bg_folder)

				# Optionally save the processed image locally before upload
				if output_folder is not None:
					save_path = output_folder / out_name
					enc_ext = Path(out_name).suffix.lower()
					ok, buf = cv2.imencode(enc_ext, result_arr)
					if ok:
						save_path.write_bytes(buf.tobytes())
					else:
						ok2, buf2 = cv2.imencode(".jpg", result_arr)
						if ok2:
							(output_folder / f"{Path(out_name).stem}.jpg").write_bytes(buf2.tobytes())
						else:
							print(f"Warning: failed to encode processed image for {img_path}, skipping save")

				upload_token = upload_data(session, out_name, data_bytes)
				try:
					create_media_item(session, upload_token, out_name, album_id)
				except requests.HTTPError as e:
					if e.response is not None and e.response.status_code in (400, 403) and album_id:
						print("[upload] album add failed, retrying library-only")
						create_media_item(session, upload_token, out_name, None)
					else:
						raise

				manifest[file_hash] = out_name
				save_manifest(manifest_path, manifest)
				print(f"[upload] uploaded: {img_path.name} -> {out_name}")
			except requests.HTTPError as http_err:
				response_text = http_err.response.text if http_err.response is not None else str(http_err)
				print(f"[upload] failed HTTP for {img_path.name}: {response_text}")
			except Exception as exc:  # pylint: disable=broad-except
				print(f"[upload] failed for {img_path.name}: {exc}")

# These wrappers are needed only for exposing exceptions neatly from the child processes.
# Basically, the check on main only shows if a process exited, with these wrappers you can traceback to the actual error.
def run_preview_process(
		folder: Path, 
		bg_folder: str, 
		window_name: str = "Preview",
		transition_time: float = 0.0,
		interval: float = UPDATE_INTERVAL_PREVIEW,
		min_age: float = MIN_AGE_PREVIEW
	) -> None:
	"""Wrapper to expose process-level exceptions with traceback."""

	try:
		preview_loop(
			folder, 
			bg_folder, 
			window_name,
			transition_time,
			interval, 
			min_age
		)
	except Exception as exc:  # pylint: disable=broad-except
		print(f"[preview] fatal error: {exc}")
		traceback.print_exc()

def run_upload_process(
		folder: Path,
		credentials: str,
		token: str,
		manifest_path: Path,
		bg_folder: str,
		album_id: str | None,
		output_folder: Path,
		interval: float,
		min_age: float
	) -> None:
	"""Wrapper to expose process-level exceptions with traceback."""

	try:
		upload_loop(
			folder,
			credentials,
			token,
			manifest_path,
			bg_folder,
			album_id,
			output_folder,
			interval,
			min_age
		)
	except Exception as exc:  # pylint: disable=broad-except
		print(f"[upload] fatal error: {exc}")
		traceback.print_exc()

def main() -> None:
	parser = argparse.ArgumentParser(description="Process images (remove green) and upload to Google Photos")
	parser.add_argument("--folder", default="images", help="Folder to scan for new images")
	parser.add_argument("--bg_folder", default="backgrounds", help="Folder containing background images (one will be chosen randomly)")
	parser.add_argument("--album_id", default="Default", 
					 	help="Optional Google Photos album ID to upload into. If None, uploads to library without album. " \
						"If 'Default', load album ID from album_id.txt (if exists)")
	parser.add_argument("--output_folder", default=None, 
					 	help="Folder to save processed images locally. If None (default) processed images won't be saved.")
	parser.add_argument("--no_preview", action="store_true", help="Disable real-time preview window")
	parser.add_argument("--preview_name", default="Preview", help="Window name for preview")
	parser.add_argument("--transition_time", type=float, default=0.0, 
					 	help="Seconds for fade transition in preview. If 0 (default) no transition will be applied.")
	parser.add_argument("--credentials", default="credentials.json", help="Path to OAuth client credentials JSON")
	parser.add_argument("--token", default="token.json", help="Path to stored user token")
	parser.add_argument("--manifest", default="manifest.json", help="Path to manifest JSON tracking uploaded files")
	parser.add_argument("--interval", type=float, default=5.0, help="Seconds between scans for new images to upload")
	parser.add_argument("--min_age", type=float, default=2.0, 
					 	help="Minimum file age (seconds) to consider a file stable before processing and uploading")
	args = parser.parse_args()

	folder = Path(args.folder).expanduser().resolve()
	manifest_path = Path(args.manifest).expanduser().resolve()
	output_folder = Path(args.output_folder).expanduser().resolve() if args.output_folder else None

	if args.album_id == "Default":
		if Path("album_id.txt").exists():
			args.album_id = Path("album_id.txt").read_text(encoding="utf-8").strip()
		else:
			print("Warning: album_id set to 'Default' but album_id.txt not found. Uploading to library without album.")
			args.album_id = None

	if not folder.exists() or not folder.is_dir():
		raise SystemExit(f"Folder not found: {folder}")
	if not os.path.exists(args.credentials):
		raise SystemExit(
			f"Credentials file not found: {args.credentials}. Download OAuth client credentials from Google Cloud Console."
		)

	if args.no_preview:
		print("Starting upload loop without preview...")
		print("Press Ctrl+C to stop.")
		try:
			upload_loop(
				folder,
				args.credentials,
				args.token,
				manifest_path,
				args.bg_folder,
				args.album_id,
				output_folder,
				args.interval,
				args.min_age
			)
		except KeyboardInterrupt:
			print("\nStopping upload loop...")
		print("Done.")
		return

	preview_process = mp.Process(
		target=run_preview_process,
		args=(folder, args.bg_folder, args.preview_name, args.transition_time),
		daemon=False,
	)
	upload_process = mp.Process(
		target=run_upload_process,
		args=(
			folder,
			args.credentials,
			args.token,
			manifest_path,
			args.bg_folder,
			args.album_id,
			output_folder,
			args.interval,
			args.min_age,
		),
		daemon=False,
	)
	preview_process.start()
	upload_process.start()
	print(f"Started processes | preview_pid={preview_process.pid} upload_pid={upload_process.pid}")
	print("Press Ctrl+C to stop.")

	try:
		while True:
			if not preview_process.is_alive():
				print(f"[main] preview process exited with code {preview_process.exitcode}")
				break
			if not upload_process.is_alive():
				print(f"[main] upload process exited with code {upload_process.exitcode}")
				break
			time.sleep(1)
	except KeyboardInterrupt:
		print("\nStopping processes...")

	if preview_process.is_alive():
		preview_process.terminate()
	if upload_process.is_alive():
		upload_process.terminate()
	preview_process.join(timeout=3)
	upload_process.join(timeout=3)
	print("Done.")

if __name__ == "__main__":
	main()
