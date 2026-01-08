#!/usr/bin/env python3
"""
Real-time log monitoring script for SMS and call activities
"""
import os
import sys
import time
import subprocess
from datetime import datetime

def monitor_logs():
    """Monitor service logs in real-time with filtering"""
    log_file = "service.log"
    
    if not os.path.exists(log_file):
        print(f"❌ Log file {log_file} not found!")
        print("Make sure the service is running and creating logs.")
        return
    
    print("🔍 Starting real-time log monitoring...")
    print("📋 Filtering for SMS, call, and error activities")
    print("Press Ctrl+C to stop\n")
    
    # Keywords to highlight
    sms_keywords = ["SMS", "📨", "📞", "Twilio", "Message SID"]
    call_keywords = ["Call", "📞", "Ultravox", "ended", "started"]
    error_keywords = ["❌", "ERROR", "Failed", "Exception"]
    
    try:
        # Use tail -f to follow the log file
        process = subprocess.Popen(
            ["tail", "-f", log_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        for line in iter(process.stdout.readline, ''):
            if line:
                line = line.strip()
                
                # Add timestamp if not present
                if not line.startswith('20'):
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    line = f"[{timestamp}] {line}"
                
                # Color coding based on content
                should_show = False
                
                # Check for SMS-related activity
                if any(keyword in line for keyword in sms_keywords):
                    print(f"\033[92m{line}\033[0m")  # Green for SMS
                    should_show = True
                
                # Check for call-related activity
                elif any(keyword in line for keyword in call_keywords):
                    print(f"\033[94m{line}\033[0m")  # Blue for calls
                    should_show = True
                
                # Check for errors
                elif any(keyword in line for keyword in error_keywords):
                    print(f"\033[91m{line}\033[0m")  # Red for errors
                    should_show = True
                
                # Show important info messages
                elif "INFO" in line and ("📊" in line or "🚀" in line or "✅" in line):
                    print(f"\033[93m{line}\033[0m")  # Yellow for info
                    should_show = True
                
                # Show all other lines in default color if they contain emojis (likely important)
                elif any(emoji in line for emoji in ["📋", "🔍", "💰", "⚠️", "🧪"]):
                    print(line)
                    should_show = True
                
                # Flush output for real-time display
                if should_show:
                    sys.stdout.flush()
                    
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped by user")
        process.terminate()
    except Exception as e:
        print(f"\n❌ Error monitoring logs: {e}")
    finally:
        if 'process' in locals():
            process.terminate()

def show_recent_sms_activity():
    """Show recent SMS-related log entries"""
    log_file = "service.log"
    
    if not os.path.exists(log_file):
        print(f"❌ Log file {log_file} not found!")
        return
    
    print("📨 Recent SMS Activity (last 50 lines):")
    print("=" * 60)
    
    try:
        # Get last 100 lines and filter for SMS activity
        result = subprocess.run(
            ["tail", "-100", log_file],
            capture_output=True,
            text=True
        )
        
        sms_lines = []
        for line in result.stdout.split('\n'):
            if any(keyword in line for keyword in ["SMS", "📨", "Twilio", "Message SID", "send_call_summary"]):
                sms_lines.append(line.strip())
        
        if sms_lines:
            for line in sms_lines[-20:]:  # Show last 20 SMS-related lines
                print(line)
        else:
            print("No recent SMS activity found in logs")
            
    except Exception as e:
        print(f"❌ Error reading logs: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--recent":
            show_recent_sms_activity()
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python monitor_logs.py           # Real-time monitoring")
            print("  python monitor_logs.py --recent  # Show recent SMS activity")
            print("  python monitor_logs.py --help    # Show this help")
        else:
            print("Unknown option. Use --help for usage information.")
    else:
        monitor_logs()