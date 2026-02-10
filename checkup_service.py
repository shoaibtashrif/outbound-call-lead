import os
import json
import logging
import requests
from urllib.parse import urlparse
import googlemaps
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session
from models import BusinessCheckup

logger = logging.getLogger(__name__)

PAGESPEED_API_URL = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"

class BusinessCheckupService:
    def __init__(self, db: Session, user_id: int):
        self.db = db
        self.user_id = user_id
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            logger.error("GOOGLE_API_KEY not found in environment")
            self.gmaps = None
        else:
            try:
                self.gmaps = googlemaps.Client(key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Google Maps client: {e}")
                self.gmaps = None

    def search_business(self, business_name, business_address=None, city_state=None):
        """
        Pulls business data from Google Places API.
        Returns enriched report with website performance and presence checks.
        """
        if not self.gmaps:
            return {"error": "Google API disabled (Missing Key)"}

        search_query = business_name
        if business_address:
            search_query = f"{business_name}, {business_address}"
        elif city_state:
            search_query = f"{business_name}, {city_state}"

        logger.info(f"Searching for business: {search_query}")
        
        try:
            # 1. Find Place request to get the Place ID
            find_place_result = self.gmaps.find_place(
                input=search_query,
                input_type="textquery",
                fields=["place_id", "name", "formatted_address"]
            )
            
            if not find_place_result or not find_place_result.get("candidates"):
                return {"error": "Business not found"}
            
            candidate = find_place_result["candidates"][0]
            place_id = candidate["place_id"]
            logger.info(f"Found candidate: {candidate.get('name')} (ID: {place_id})")
            
            # 2. Place Details request using the Place ID
            place_details = self.gmaps.place(
                place_id=place_id,
                fields=[
                    "name", "type", "rating", "user_ratings_total",
                    "opening_hours", "photo", "reviews",
                    "business_status", "formatted_address", "url",
                    "formatted_phone_number", "website"
                ]
            )
            
            if not place_details or not place_details.get("result"):
                return {"error": "Could not retrieve detailed Place information"}
            
            result = place_details["result"]
            
            # Fetch actual photo URLs
            photos_urls = []
            if "photo" in result:
                for photo_ref_obj in result["photo"][:3]:  # Limit to 3 photos
                    try:
                        photo_url = self.gmaps.places_photo(photo_ref_obj["photo_reference"], max_width=800)
                        photos_urls.append(photo_url)
                    except Exception as photo_e:
                        logger.warning(f"Error fetching photo URL: {photo_e}")
                        pass
            
            website_url = result.get("website")
            
            # Build the enriched report
            report = self._build_report(result, photos_urls, website_url)
            
            return report

        except Exception as e:
            logger.error(f"Error searching business: {e}")
            return {"error": str(e)}

    def _build_report(self, place_data, photos_urls, website_url):
        """
        Builds a comprehensive report from Place data, PageSpeed, and website checks.
        """
        # Extract primary category (type)
        primary_category = None
        if "type" in place_data and place_data["type"]:
            primary_category = place_data["type"][0] if isinstance(place_data["type"], list) else place_data["type"]
        
        # Extract opening hours
        opening_hours_data = None
        if "opening_hours" in place_data and place_data["opening_hours"]:
            opening_hours_obj = place_data["opening_hours"]
            # Get weekday_text which has formatted hours like "Monday: 9:00 AM – 5:00 PM"
            opening_hours_data = opening_hours_obj.get("weekday_text", [])
        
        # Extract reviews
        reviews_data = []
        if "reviews" in place_data and place_data["reviews"]:
            for review in place_data["reviews"][:5]:  # Limit to 5 reviews
                reviews_data.append({
                    "author_name": review.get("author_name", "Anonymous"),
                    "rating": review.get("rating", 0),
                    "text": review.get("text", ""),
                    "time": review.get("time", 0),
                    "relative_time": review.get("relative_time_description", "")
                })
        
        # Basic business details
        business_details = {
            "name": place_data.get("name"),
            "address": place_data.get("formatted_address"),
            "phone": place_data.get("formatted_phone_number"),
            "website": website_url,
            "rating": place_data.get("rating"),
            "review_count": place_data.get("user_ratings_total"),
            "status": place_data.get("business_status"),
            "google_maps_url": place_data.get("url"),
            "primary_category": primary_category,
            "photos": photos_urls,
            "opening_hours": opening_hours_data,
            "reviews": reviews_data
        }
        
        # Completeness check
        completeness_check = {
            "has_website": bool(website_url),
            "has_phone": bool(place_data.get("formatted_phone_number")),
            "has_hours": "opening_hours" in place_data and place_data["opening_hours"],
            "has_photos": bool(photos_urls),
            "has_reviews": place_data.get("user_ratings_total", 0) > 0,
            "rating_health": "Good" if (place_data.get("rating") or 0) >= 4.0 else "Needs Improvement"
        }
        
        # Website performance and presence checks
        website_performance = {}
        presence_checks = {}
        
        if website_url:
            website_performance = self._get_pagespeed_insights(website_url)
            presence_checks = self._perform_website_presence_checks(website_url)
        else:
            website_performance = {"note": "No website available"}
            presence_checks = {"note": "No website available"}
        
        report = {
            "business_details": business_details,
            "completeness_check": completeness_check,
            "website_performance": website_performance,
            "presence_checks": presence_checks
        }
        
        return report

    def _get_pagespeed_insights(self, website_url):
        """
        Pulls website performance data using Google PageSpeed Insights API.
        """
        pagespeed_data = {
            "mobile_score": 0,
            "desktop_score": 0,
            "core_web_vitals_summary": "N/A"
        }
        
        try:
            # Mobile Score
            logger.info(f"Getting PageSpeed Insights for Mobile ({website_url})")
            mobile_params = {
                "url": website_url,
                "key": self.api_key,
                "strategy": "MOBILE"
            }
            mobile_response = requests.get(PAGESPEED_API_URL, params=mobile_params, timeout=30)
            
            if mobile_response.status_code == 200:
                mobile_data = mobile_response.json()
                lighthouse = mobile_data.get("lighthouseResult", {})
                categories = lighthouse.get("categories", {})
                pagespeed_data["mobile_score"] = int(categories.get("performance", {}).get("score", 0) * 100)
                
                # Core Web Vitals
                if "loadingExperience" in mobile_data and "metrics" in mobile_data["loadingExperience"]:
                    cwv_metrics = mobile_data["loadingExperience"]["metrics"]
                    is_passing = True
                    primary_cwv_keys = ["LARGEST_CONTENTFUL_PAINT_MS", "CUMULATIVE_LAYOUT_SHIFT_SCORE", 
                                       "FIRST_INPUT_DELAY_MS", "INTERACTION_TO_NEXT_PAINT_MS"]
                    
                    relevant_cwv_results = {k: v for k, v in cwv_metrics.items() if k in primary_cwv_keys}
                    
                    if relevant_cwv_results:
                        for metric_key, metric_data in relevant_cwv_results.items():
                            if metric_data.get('category') != 'GOOD':
                                is_passing = False
                                break
                        pagespeed_data["core_web_vitals_summary"] = "PASS" if is_passing else "NEEDS_IMPROVEMENT"
                    else:
                        pagespeed_data["core_web_vitals_summary"] = "NO_CWV_DATA"
            else:
                logger.warning(f"PageSpeed Mobile error: {mobile_response.status_code}")
            
            # Desktop Score
            logger.info(f"Getting PageSpeed Insights for Desktop ({website_url})")
            desktop_params = {
                "url": website_url,
                "key": self.api_key,
                "strategy": "DESKTOP"
            }
            desktop_response = requests.get(PAGESPEED_API_URL, params=desktop_params, timeout=30)
            
            if desktop_response.status_code == 200:
                desktop_data = desktop_response.json()
                lighthouse = desktop_data.get("lighthouseResult", {})
                categories = lighthouse.get("categories", {})
                pagespeed_data["desktop_score"] = int(categories.get("performance", {}).get("score", 0) * 100)
            else:
                logger.warning(f"PageSpeed Desktop error: {desktop_response.status_code}")
                
        except Exception as e:
            logger.error(f"PageSpeed check failed: {e}")
        
        return pagespeed_data

    def _perform_website_presence_checks(self, website_url):
        """
        Performs basic website presence checks.
        """
        if not website_url.startswith(('http://', 'https://')):
            website_url = 'https://' + website_url
        
        checks = {
            "https_enabled": False,
            "mobile_responsive": False,
            "contact_form_present": False,
            "click_to_call_detected": False,
            "error": None
        }
        
        try:
            response = requests.get(website_url, timeout=10, allow_redirects=True)
            response.raise_for_status()
            
            final_url = response.url
            parsed_url = urlparse(final_url)
            checks["https_enabled"] = parsed_url.scheme.lower() == "https"
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Mobile Responsive (simplified - viewport meta tag)
            if soup.find('meta', {'name': 'viewport', 'content': lambda x: x and 'width=device-width' in x}):
                checks["mobile_responsive"] = True
            
            # Contact form present
            if soup.find('form', action=lambda x: x and ('contact' in x.lower() or 'submit' in x.lower())) or \
               soup.find(lambda tag: tag.name == 'a' and tag.get_text() and ('contact' in tag.get_text().lower() or 'get in touch' in tag.get_text().lower())):
                checks["contact_form_present"] = True
            
            # Click-to-call detected
            if soup.find('a', href=lambda href: href and href.startswith('tel:')):
                checks["click_to_call_detected"] = True
                
        except requests.exceptions.Timeout:
            logger.warning(f"Website check timed out for {website_url}")
            checks["error"] = "Website check timed out"
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error accessing website {website_url}: {e}")
            checks["error"] = f"Error accessing website: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error during website check: {e}")
            checks["error"] = f"Unexpected error: {str(e)}"
        
        return checks

    def save_report(self, report_data: dict, is_starred=True):
        details = report_data.get("business_details", {})
        
        new_checkup = BusinessCheckup(
            user_id=self.user_id,
            business_name=details.get("name", "Unknown"),
            address=details.get("address"),
            website=details.get("website"),
            phone=details.get("phone"),
            email=None,
            report_data=json.dumps(report_data),
            is_starred=is_starred
        )
        self.db.add(new_checkup)
        self.db.commit()
        self.db.refresh(new_checkup)
        return new_checkup
    
    def list_starred_reports(self, skip=0, limit=50):
        return self.db.query(BusinessCheckup).filter(
            BusinessCheckup.user_id == self.user_id,
            BusinessCheckup.is_starred == True
        ).order_by(BusinessCheckup.created_at.desc()).offset(skip).limit(limit).all()

    def get_report(self, checkup_id: int):
        r = self.db.query(BusinessCheckup).filter(
            BusinessCheckup.id == checkup_id,
            BusinessCheckup.user_id == self.user_id
        ).first()
        return r
    
    def delete_report(self, checkup_id: int):
        r = self.get_report(checkup_id)
        if r:
            self.db.delete(r)
            self.db.commit()
            return True
        return False
