import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from typing import List, Dict, Any, Optional

class GoogleSheetsService:
    def __init__(self):
        self.scopes = ['https://www.googleapis.com/auth/spreadsheets']
        self.service_account_file = os.path.join(os.path.dirname(__file__), 'service-account.json')
        self._service = None

    @property
    def service(self):
        if self._service is None:
            if not os.path.exists(self.service_account_file):
                print(f"⚠️ Google Service Account file not found at {self.service_account_file}")
                return None
            
            try:
                creds = service_account.Credentials.from_service_account_file(
                    self.service_account_file, scopes=self.scopes)
                self._service = build('sheets', 'v4', credentials=creds)
            except Exception as e:
                print(f"❌ Error initializing Google Sheets service: {e}")
                return None
        return self._service

    def append_row(self, spreadsheet_id: str, range_name: str, values: List[Any]):
        """
        Append a row of values to a spreadsheet.
        range_name is usually the sheet name (e.g., 'Sheet1')
        """
        if not self.service:
            return {"success": False, "error": "Google Sheets service not initialized"}

        try:
            body = {
                'values': [values]
            }
            result = self.service.spreadsheets().values().append(
                spreadsheetId=spreadsheet_id,
                range=range_name,
                valueInputOption='RAW',
                insertDataOption='INSERT_ROWS',
                body=body
            ).execute()
            
            return {"success": True, "updated_range": result.get('updates', {}).get('updatedRange')}
        except Exception as e:
            print(f"❌ Error appending row to Google Sheet: {e}")
            return {"success": False, "error": str(e)}

    def append_data_dict(self, spreadsheet_id: str, sheet_name: str, data: Dict[str, Any]):
        """
        Append data from a dictionary. 
        If the sheet is empty, it will write headers first.
        """
        if not self.service:
            return {"success": False, "error": "Google Sheets service not initialized"}

        try:
            # 1. Get existing headers or check if sheet is empty
            result = self.service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=f"{sheet_name}!1:1"
            ).execute()
            
            existing_headers = result.get('values', [[]])[0]
            
            if not existing_headers:
                # Sheet is empty, write headers first
                headers = list(data.keys())
                self.append_row(spreadsheet_id, sheet_name, headers)
                existing_headers = headers
            
            # 2. Map data to headers
            row_values = []
            for header in existing_headers:
                row_values.append(data.get(header, ""))
            
            # 3. Append the row
            return self.append_row(spreadsheet_id, sheet_name, row_values)
            
        except Exception as e:
            print(f"❌ Error in append_data_dict: {e}")
            return {"success": False, "error": str(e)}

google_sheets_service = GoogleSheetsService()
