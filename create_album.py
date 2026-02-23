import os
import argparse
from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


parser = argparse.ArgumentParser(description="Create a new Google Photos album")
parser.add_argument("album_title", type=str, help="Title of the new album")
args = parser.parse_args()

title = args.album_title

SCOPES = ["https://www.googleapis.com/auth/photoslibrary.appendonly"]

flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
creds = flow.run_local_server(port=0)

service = build("photoslibrary", "v1", credentials=creds, static_discovery=False)

album_body = {
    "album": {"title": title}
}

response = service.albums().create(body=album_body).execute()
print(f"Created album with title: {response['title']} and id: {response['id']}")

# Save album ID to a local file for future reference
with open("album_id.txt", "w", encoding="utf-8") as f:
    f.write(response["id"])
print("Album ID saved to album_id.txt")