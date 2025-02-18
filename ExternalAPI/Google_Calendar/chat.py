import os                                                           #to interact with OS 
import pickle                                                       #saving authentication credentials
import datetime
import streamlit as st
from google.oauth2.credentials import Credentials                   #manages user authentication credentials 
from google_auth_oauthlib.flow import InstalledAppFlow              #authorization flow for installed apps
from google.auth.transport.requests import Request                  #to refresh expired tokens 
from googleapiclient.discovery import build                         #to interact with google APIs

# Set up API scope (Read & Write Access)
SCOPES = ['https://www.googleapis.com/auth/calendar']

# Authentication function
def authenticate_google():
    creds = None
    token_file = 'token.pickle'

    # Load existing credentials
    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)

    # Refresh or re-authenticate if needed
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)    #to start Oauth 2.0 authorization + user prompted to log in via google 
            creds = flow.run_local_server(port=0)                                           #starts a temporary local web server where the user can grant permissions.

        # Save credentials
        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)

    return build('calendar', 'v3', credentials=creds)

# Initialize Streamlit UI
st.set_page_config(page_title="Google Calendar Manager", layout="wide")
st.title("📅 Google Calendar Manager")

# Authenticate Google Calendar
service = authenticate_google()

# Function to list events
def list_events():
    now = datetime.datetime.utcnow().isoformat() + 'Z'
    events_result = service.events().list(
        calendarId='primary', timeMin=now, maxResults=20, singleEvents=True, orderBy='startTime'
    ).execute()
    
    events = events_result.get('items', [])
    return events

# Function to create an event
def create_event(summary, location, description, start_datetime, end_datetime):
    event = {
        'summary': summary,
        'location': location,
        'description': description,
        'start': {'dateTime': start_datetime, 'timeZone': 'Asia/Kolkata'},
        'end': {'dateTime': end_datetime, 'timeZone': 'Asia/Kolkata'},
    }
    event = service.events().insert(calendarId='primary', body=event).execute()
    return event.get('htmlLink')

# Function to update an event
def update_event(event_id, summary, description, start_datetime, end_datetime):
    try:
        event = service.events().get(calendarId='primary', eventId=event_id).execute()
        event['summary'] = summary
        event['description'] = description
        event['start']['dateTime'] = start_datetime
        event['end']['dateTime'] = end_datetime
        updated_event = service.events().update(calendarId='primary', eventId=event_id, body=event).execute()
        return updated_event.get('htmlLink')
    except Exception as e:
        return f"Error updating event: {e}"

# Function to delete an event
def delete_event(event_id):
    try:
        service.events().delete(calendarId='primary', eventId=event_id).execute()
        return "Event deleted successfully."
    except Exception as e:
        return f"Error deleting event: {e}"

# Sidebar Navigation
menu = st.sidebar.radio("Choose an option", ["List Events", "Create Event", "Update Event", "Delete Event"])

if menu == "List Events":
    st.subheader(" \U0001F4C6 Upcoming Events")
    events = list_events()
    if events:
        event_data = [{"Start": e['start'].get('dateTime', e['start'].get('date')), 
                       "Summary": e.get('summary', 'No Title'), 
                       "Event ID": e['id']} for e in events]
        st.table(event_data)
    else:
        st.write("No upcoming events found.")

elif menu == "Create Event":
    st.subheader(" \U0001F195 Create a New Event")
    with st.form("create_event_form"):
        summary = st.text_input("Event Title", "Sample Event")
        location = st.text_input("Location", "123 Sample St, Sample City")
        description = st.text_area("Description", "This is a sample event.")
        start_datetime = st.text_input("Start Date & Time (YYYY-MM-DDTHH:MM:SS)", "2025-02-28T09:00:00")
        end_datetime = st.text_input("End Date & Time (YYYY-MM-DDTHH:MM:SS)", "2025-02-28T17:00:00")
        submit = st.form_submit_button("Create Event")

        if submit:
            event_link = create_event(summary, location, description, start_datetime, end_datetime)
            st.success(f"✅ Event created! [View it here]({event_link})")

elif menu == "Update Event":
    st.subheader(" \U00002712 Update an Existing Event")
    with st.form("update_event_form"):
        event_id = st.text_input("Event ID")
        summary = st.text_input("New Event Title", "Updated Event")
        description = st.text_area("New Description", "Updated event description.")
        start_datetime = st.text_input("New Start Date & Time (YYYY-MM-DDTHH:MM:SS)", "2025-02-28T10:00:00")
        end_datetime = st.text_input("New End Date & Time (YYYY-MM-DDTHH:MM:SS)", "2025-02-28T18:00:00")
        submit = st.form_submit_button("Update Event")

        if submit:
            result = update_event(event_id, summary, description, start_datetime, end_datetime)
            st.success(f"✅ Event updated! [View it here]({result})")

elif menu == "Delete Event":
    st.subheader(" \U0000274E Delete an Event")
    with st.form("delete_event_form"):
        event_id = st.text_input("Event ID")
        submit = st.form_submit_button("Delete Event")

        if submit:
            result = delete_event(event_id)
            st.success(result)
