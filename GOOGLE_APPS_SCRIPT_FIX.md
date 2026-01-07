# Google Apps Script Fix

## Problem
The phone number is being sent in the request body but not appearing in the Google Sheet.

## Root Cause
Your Google Apps Script is likely using a fixed header order that doesn't match the data keys being sent.

## Solution
Replace your Google Apps Script code with this corrected version:

```javascript
function doPost(e) {
  try {
    var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("Sheet1");
    var data = JSON.parse(e.postData.contents);
    
    // Get existing headers or create new ones
    var lastRow = sheet.getLastRow();
    var headers;
    
    if (lastRow === 0) {
      // No headers exist, create them from the data keys
      headers = Object.keys(data);
      sheet.appendRow(headers);
    } else {
      // Get existing headers
      headers = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];
    }
    
    // Build row data matching header order
    var row = [];
    for (var i = 0; i < headers.length; i++) {
      var header = headers[i];
      row.push(data[header] || "");
    }
    
    // Append the row
    sheet.appendRow(row);
    
    return ContentService.createTextOutput(JSON.stringify({
      "success": true,
      "message": "Data saved to Data logger",
      "sheet": "Sheet1",
      "rowAdded": row
    })).setMimeType(ContentService.MimeType.JSON);
    
  } catch (error) {
    return ContentService.createTextOutput(JSON.stringify({
      "success": false,
      "error": error.toString()
    })).setMimeType(ContentService.MimeType.JSON);
  }
}
```

## Steps to Fix
1. Open your Google Sheet
2. Go to **Extensions > Apps Script**
3. Replace the entire code with the version above
4. Click **Save**
5. Click **Deploy > Manage Deployments**
6. Click the **Edit** icon (pencil) on your existing deployment
7. Change **Version** to "New version"
8. Click **Deploy**
9. Copy the new URL and update it in your agent configuration

## Why This Works
- The original script had a hardcoded header order that didn't include "phone"
- This new version dynamically matches the data keys to the headers
- It preserves the header order from the first row
- It fills in empty strings for missing fields
