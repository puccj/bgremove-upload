import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Tuple

from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Google Photos scope for adding media items (no read/delete)
SCOPES = ["https://www.googleapis.com/auth/photoslibrary.appendonly"]

# class Uploader:
# 	def __init__(self, credentials_path: str, token_path: str, manifest_path: str, album_id: str | None = None):
# 		# Get session
# 		creds = load_credentials(credentials_path, token_path)
# 		self.session = AuthorizedSession(creds)
# 		self.manifest_path = Path(manifest_path)
# 		self.manifest = load_manifest(self.manifest_path)
# 		self.album_id = album_id
# 		self.hach_cache: Dict[Path, Tuple[int, int, str]] = {}

def get_session(credentials_path: str, token_path: str) -> AuthorizedSession:
	creds = load_credentials(credentials_path, token_path)
	return AuthorizedSession(creds)

def load_manifest(manifest_path: Path) -> Dict[str, str]:
	"""Load manifest mapping file hash to uploaded filename."""
	if manifest_path.exists():
		with manifest_path.open("r", encoding="utf-8") as fp:
			return json.load(fp)
	return {}

def save_manifest(manifest_path: Path, manifest: Dict[str, str]) -> None:
	manifest_path.parent.mkdir(parents=True, exist_ok=True)
	with manifest_path.open("w", encoding="utf-8") as fp:
		json.dump(manifest, fp, indent=2)

def hash_file(path: Path) -> str:
	"""Compute SHA-256 hash of a file."""
	sha = hashlib.sha256()
	with path.open("rb") as fp:
		for chunk in iter(lambda: fp.read(8192), b""):
			sha.update(chunk)
	return sha.hexdigest()

def load_credentials(credentials_path: str, token_path: str) -> Credentials:
	"""Load or create OAuth credentials for Google Photos."""
	creds = None
	if os.path.exists(token_path):
		creds = Credentials.from_authorized_user_file(token_path, SCOPES)
	if not creds or not creds.valid:
		if creds and creds.expired and creds.refresh_token:
			creds.refresh(Request())
		else:
			flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
			creds = flow.run_local_server(port=0)
		with open(token_path, "w", encoding="utf-8") as token_file:
			token_file.write(creds.to_json())
	return creds

def upload_data(session: AuthorizedSession, file_name: str, data: bytes) -> str:
	"""Upload provided bytes with a given filename to Google Photos and return upload token."""
	headers = {
		"Content-type": "application/octet-stream",
		"X-Goog-Upload-File-Name": file_name,
		"X-Goog-Upload-Protocol": "raw",
	}
	resp = session.post("https://photoslibrary.googleapis.com/v1/uploads", data=data, headers=headers)
	resp.raise_for_status()
	return resp.text

def create_media_item(session: AuthorizedSession, upload_token: str, description: str, album_id: str | None) -> None:
	payload = {
		"newMediaItems": [
			{
				"description": description,
				"simpleMediaItem": {"uploadToken": upload_token},
			}
		]
	}
	if album_id:
		payload["albumId"] = album_id
	resp = session.post("https://photoslibrary.googleapis.com/v1/mediaItems:batchCreate", json=payload)
	resp.raise_for_status()
	body = resp.json()
	status = body.get("newMediaItemResults", [{}])[0].get("status", {})
	if status.get("code") not in (None, 0):
		message = status.get("message", "Unknown error")
		raise RuntimeError(f"Batch create failed: {message}")