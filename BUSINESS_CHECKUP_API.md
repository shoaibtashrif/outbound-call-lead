# Business Checkup API Documentation

Specifically designed for automated business auditing and lead generation. This API provides deterministic visibility scoring and revenue leakage estimates based on Google Places data and Website performance metrics.

## Endpoint

**URL:** `http://localhost:8002/api/business-checkup/search`  
**Method:** `POST`  
**Auth:** Optional (Publicly accessible)

---

## 1. Request Schema

```json
{
  "full_name": "John Doe",           // Required
  "phone": "+1234567890",            // Required
  "email": "john@example.com",       // Required
  "business_name": "McDonalds",      // Required
  "city_state": "New York, NY",      // Required (for disambiguation)
  "business_address": "string",       // Optional
  "website": "string"                // Optional
}
```

---

## 2. Response Schema

The response is divided into four main blocks:

### `business_info`
*   `name`: Registered business name.
*   `address`: Formatted physical address.
*   `phone`: Primary phone number.
*   `website`: URL of the business website.

### `google_metrics`
*   `rating`: Google Maps star rating (0.0 - 5.0).
*   `review_count`: Total number of user reviews.
*   `category`: Primary business category (e.g., `cafe`).
*   `verification_status`: Operational status.
*   `has_photos`: Boolean (True/False) indicating photo presence.
*   `profile_completeness`: Flags for website, phone, hours, and photos.

### `website_metrics`
*   `mobile_score`: Google PageSpeed Performance score (0-100).
*   `desktop_score`: Google PageSpeed Performance score (0-100).
*   `core_web_vitals_summary`: `PASS`, `NEEDS_IMPROVEMENT`, or `FAIL`.
*   `https_enabled`: Boolean indicating if the site is secure.

### `calculated_scores`
*   `visibility_score`: Total score (0-100) based on visibility weighting.
*   `estimated_loss_percentage`: % of potential customers lost.
*   `estimated_monthly_revenue_leakage`: Dollar value calculation.

---

## 3. Calculation Formulas (Deterministic)

### Business Visibility Score (100 pts Total)
| Category | Metric | Points |
| :--- | :--- | :--- |
| **Google Presence** | Rating 4.5+ | 20 pts |
| | Review Volume (50+) | 10 pts |
| | Profile Completeness | 20 pts (5 each: Web, Phone, Hours, Photos) |
| **Website Performance** | Mobile Speed (90+) | 20 pts |
| | Desktop Speed (90+) | 10 pts |
| | HTTPS Enabled | 10 pts |
| **Trust Signals** | Core Web Vitals (PASS) | 10 pts |

### Revenue Leakage Logic
*   **Multiplier:** 100 leads/month × $1,500 average job value.
*   **Loss Factors (Stackable up to 40%):**
    *   Rating < 4.0: **-15%**
    *   Mobile Score < 70: **-10%**
    *   Review Count < 10: **-10%**
    *   No Photos: **-5%**

---

## 4. Sample Request & Response

### Sample cURL Command
```bash
curl -X 'POST' 'http://localhost:8002/api/business-checkup/search' \
  -H 'Content-Type: application/json' \
  -d '{
    "full_name": "John Doe",
    "phone": "+1-555-0199",
    "business_name": "McDonalds Times Square",
    "email": "john@example.com",
    "city_state": "New York, NY"
  }'
```

### Sample JSON Response
```json
{
  "business_info": {
    "name": "McDonald's",
    "address": "1528 Broadway Times Square, New York, NY 10036, USA",
    "phone": "(917) 409-5946",
    "website": "https://www.mcdonalds.com/us/en-us/..."
  },
  "google_metrics": {
    "rating": 3.7,
    "review_count": 1593,
    "category": "cafe",
    "verification_status": "OPERATIONAL",
    "has_photos": true,
    "profile_completeness": {
      "has_website": true,
      "has_phone": true,
      "has_hours": true,
      "has_photos": true
    }
  },
  "website_metrics": {
    "mobile_score": 53,
    "desktop_score": 65,
    "core_web_vitals_summary": "NEEDS_IMPROVEMENT",
    "https_enabled": true
  },
  "calculated_scores": {
    "visibility_score": 67,
    "estimated_loss_percentage": 25,
    "estimated_monthly_revenue_leakage": 37500
  }
}
```

---

## 5. Example Integration (JavaScript)

```javascript
const fetchCheckup = async () => {
  const response = await fetch('http://localhost:8002/api/business-checkup/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      full_name: "Frontend Lead",
      phone: "+1-555-0100",
      business_name: "Local Pizza Shop",
      email: "contact@pizza.com",
      city_state: "Chicago, IL"
    })
  });
  const data = await response.json();
  console.log("Visibility Score:", data.calculated_scores.visibility_score);
};
```
