import os
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import logging

logger = logging.getLogger(__name__)

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/calendar.events'
]

def get_google_credentials():
    """
    Loads the Google OAuth token.json file and refreshes it if it's expired.
    Returns the valid Credentials object or None if the token is missing/invalid.
    """
    creds = None
    token_path = os.path.join(os.path.dirname(__file__), 'token.json')
    
    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        except Exception as e:
            logger.error(f"❌ Error loading token.json: {e}")
            return None
    else:
        logger.warning(f"⚠️ Google OAuth token not found at {token_path}. Run auth_google_desktop.py first.")
        return None

    # Refresh the token if it has expired
    if creds and creds.expired and creds.refresh_token:
        try:
            logger.info("🔄 Google OAuth token expired. Refreshing token...")
            creds.refresh(Request())
            # Save the refreshed token back to token.json
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
            logger.info("✅ Google OAuth token refreshed successfully.")
        except Exception as e:
            logger.error(f"❌ Error refreshing Google OAuth token: {e}")
            return None

    return creds
