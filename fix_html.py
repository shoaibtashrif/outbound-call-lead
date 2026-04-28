import re

with open("templates/dashboard.html", "r", encoding="utf-8") as f:
    html = f.read()

# The missing sections start at:
start_find_business = html.find('<!-- FIND BUSINESS SECTION -->')

# We know the checkup section ends somewhere before <!-- Create Tool Modal -->
# Let's find exactly where section-checkup ends.
# It's an HTML div. Let's just find "<!-- Create Tool Modal -->"
create_tool_modal_idx = html.find('<!-- Create Tool Modal -->')

# The text to move
text_to_move = html[start_find_business:create_tool_modal_idx]

# Remove it from the original place
new_html = html[:start_find_business] + html[create_tool_modal_idx:]

# Now find the end of the <div class="main-content">
# main-content contains <!-- Dashboard Section -->, <!-- New Call Section -->, etc., down to <!-- SMS Inbox Section -->
# So let's find the end of <!-- SMS Inbox Section -->. 
# It ends right before <!-- Call Details Modal -->
call_details_modal_idx = new_html.find('<!-- Call Details Modal -->')

# We want to insert text_to_move right before call_details_modal_idx
final_html = new_html[:call_details_modal_idx] + text_to_move + new_html[call_details_modal_idx:]

with open("templates/dashboard.html", "w", encoding="utf-8") as f:
    f.write(final_html)
    
print("Sections successfully moved inside main-content.")
