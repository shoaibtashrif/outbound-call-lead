import datetime
from googleapiclient.discovery import build
from typing import Dict, Any, List
from google_auth_service import get_google_credentials
import logging

logger = logging.getLogger(__name__)

class GoogleCalendarService:
    def __init__(self):
        self._service = None

    @property
    def service(self):
        if self._service is None:
            creds = get_google_credentials()
            if not creds:
                logger.warning("⚠️ Google User Token file not found or invalid.")
                return None
            try:
                self._service = build('calendar', 'v3', credentials=creds)
            except Exception as e:
                logger.error(f"❌ Error initializing Google Calendar service: {e}")
                return None
        return self._service

    def get_events(self, date_str: str) -> Dict[str, Any]:
        """
        Get events for a specific date on the primary calendar.
        Format expected for date_str: YYYY-MM-DD
        """
        if not self.service:
            return {"success": False, "error": "Google Calendar service not initialized"}
        
        try:
            # Parse date string to get start and end of that day in UTC
            dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            time_min = dt.isoformat() + 'Z'
            time_max = (dt + datetime.timedelta(days=1)).isoformat() + 'Z'

            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            # Format nicely for the AI to read
            formatted_events = []
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                end = event['end'].get('dateTime', event['end'].get('date'))
                formatted_events.append({
                    "summary": event.get("summary", "Busy"),
                    "start": start,
                    "end": end
                })

            return {"success": True, "events": formatted_events, "date": date_str}

        except Exception as e:
            logger.error(f"❌ Error getting calendar events: {e}")
            return {"success": False, "error": str(e)}

    def create_event(self, summary: str, description: str, start_time: str, end_time: str) -> Dict[str, Any]:
        """
        Create a new event on the primary calendar.
        start_time and end_time must be ISO formatted strings (e.g. 2026-02-25T14:30:00Z format, 
        or with timezone offset 2026-02-25T14:30:00-05:00)
        """
        if not self.service:
            return {"success": False, "error": "Google Calendar service not initialized"}

        try:
            event_body = {
                'summary': summary,
                'description': description,
                'start': {
                    'dateTime': start_time,
                },
                'end': {
                    'dateTime': end_time,
                },
            }

            created_event = self.service.events().insert(
                calendarId='primary',
                body=event_body
            ).execute()

            return {
                "success": True, 
                "event_id": created_event.get('id'), 
                "event_link": created_event.get('htmlLink')
            }
        except Exception as e:
            logger.error(f"❌ Error creating calendar event: {e}")
            return {"success": False, "error": str(e)}

google_calendar_service = GoogleCalendarService()
