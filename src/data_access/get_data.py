import os
import re
import time

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Google Drive API Scope (Download Permission)
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def authenticate():
    """Authenticate and return Google Drive API service."""
    creds = None

    # Load existing credentials
    if os.path.exists("../../creds/token.json"):
        creds = Credentials.from_authorized_user_file("../../creds/token.json", SCOPES)
    # If credentials are invalid or missing, reauthenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "../../creds/credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Save new credentials
        with open("../../creds/token.json", "w") as token:
            token.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def extract_file_id(url):
    """Extract Google Drive File ID from URL."""
    patterns = [
        r"https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",  # Pattern 1
        r"https://drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",  # Pattern 2
        r"https://drive\.google\.com/uc\?id=([a-zA-Z0-9_-]+)",  # Pattern 3
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)  # Return extracted File ID

    return None  # Return None if no match is found


def download_video(service, file_id):
    """Download a video file from Google Drive using its File ID."""
    try:
        # Get video metadata (name)
        file_metadata = (
            service.files()
            .get(fileId=file_id, fields="name", supportsAllDrives=True)
            .execute()
        )
        file_name = file_metadata.get("name", "unknown_video.mp4")

        # Ensure the 'videos' folder exists
        os.makedirs("videos", exist_ok=True)
        file_path = os.path.join("videos", file_name)

        print(f"⬇️ Downloading: {file_name} (ID: {file_id})...")

        # Download the file
        request = service.files().get_media(fileId=file_id)
        with open(file_path, "wb") as f:
            f.write(request.execute())

        print(f"✅ Download Complete: {file_path}\n")

    except HttpError as error:
        print(f"❌ Error downloading file {file_id}: {error}")


def process_urls(service):
    """Read URLs from 'urls.txt', extract File IDs, and download videos."""
    if not os.path.exists("urls.txt"):
        print("❌ urls.txt file not found!")
        return

    with open("urls.txt", "r") as file:
        urls = file.readlines()

    for url in urls:
        url = url.strip()
        if url:
            file_id = extract_file_id(url)
            if file_id:
                download_video(service, file_id)
                time.sleep(2)
            else:
                print(f"❌ Invalid URL: {url}\n")


def main():
    """Main function to authenticate and download videos."""
    service = authenticate()
    process_urls(service)


if __name__ == "__main__":
    main()


# from google_auth_oauthlib.flow import InstalledAppFlow
# import json
#
# SCOPES = ["https://www.googleapis.com/auth/drive"]
#
# def authenticate():
#     flow = InstalledAppFlow.from_client_secrets_file(
#         "token.json", SCOPES
#     )
#     creds = flow.run_local_server(port=0)
#
#     # Save the credentials to token.json
#     with open("token.json", "w") as token_file:
#         token_file.write(creds.to_json())
#
#     return creds
#
# creds = authenticate()
#
# # Print the refresh token (for verification)
# print("Refresh Token:", creds.refresh_token)
#
