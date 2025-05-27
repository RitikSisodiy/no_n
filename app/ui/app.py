import streamlit as st
import pandas as pd
from datetime import datetime
import json
import requests
from typing import Dict, Any, Optional
import os
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import numpy as np
import asyncio
import websockets
import threading
import queue
import uuid

# Configure the page
st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide"
)

# Constants
API_URL = os.getenv("API_URL", "http://localhost:8000")
WS_URL = "ws://localhost:8000/ws"

# Session state initialization
if "token" not in st.session_state:
    st.session_state.token = None
if "user" not in st.session_state:
    st.session_state.user = None
if "ws_connected" not in st.session_state:
    st.session_state.ws_connected = False
if "notifications" not in st.session_state:
    st.session_state.notifications = []
if "document_subscriptions" not in st.session_state:
    st.session_state.document_subscriptions = set()
if "user_subscriptions" not in st.session_state:
    st.session_state.user_subscriptions = set()

# WebSocket message queue
message_queue = queue.Queue()

def get_auth_headers():
    """Get authentication headers for API requests."""
    return {
        "Authorization": f"Bearer {st.session_state.token}"
    } if st.session_state.token else {}

async def websocket_handler():
    """Handle WebSocket connection and messages."""
    while True:
        try:
            if not st.session_state.token or not st.session_state.user:
                await asyncio.sleep(1)
                continue
            
            async with websockets.connect(f"{WS_URL}/{st.session_state.user['id']}") as websocket:
                st.session_state.ws_connected = True
                
                # Subscribe to documents
                for doc_id in st.session_state.document_subscriptions:
                    await websocket.send(json.dumps({
                        "action": "subscribe_document",
                        "document_id": doc_id
                    }))
                
                # Subscribe to users
                for user_id in st.session_state.user_subscriptions:
                    await websocket.send(json.dumps({
                        "action": "subscribe_user",
                        "user_id": user_id
                    }))
                
                while True:
                    try:
                        message = await websocket.recv()
                        message_queue.put(json.loads(message))
                    except websockets.exceptions.ConnectionClosed:
                        break
                    except Exception as e:
                        st.error(f"WebSocket error: {str(e)}")
                        break
                
        except Exception as e:
            st.error(f"WebSocket connection error: {str(e)}")
            st.session_state.ws_connected = False
            await asyncio.sleep(5)  # Wait before reconnecting

def start_websocket_thread():
    """Start WebSocket handler in a separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(websocket_handler())

def process_websocket_messages():
    """Process WebSocket messages from the queue."""
    while not message_queue.empty():
        try:
            message = message_queue.get_nowait()
            handle_websocket_message(message)
        except queue.Empty:
            break
        except Exception as e:
            st.error(f"Error processing WebSocket message: {str(e)}")

def handle_websocket_message(message: Dict[str, Any]):
    """Handle incoming WebSocket messages."""
    message_type = message.get("type")
    
    if message_type == "document_update":
        handle_document_update(message)
    elif message_type == "user_update":
        handle_user_update(message)
    elif message_type == "system_update":
        handle_system_update(message)
    elif message_type == "error":
        st.error(f"WebSocket error: {message.get('message')}")

def handle_document_update(message: Dict[str, Any]):
    """Handle document update messages."""
    update_type = message.get("update_type")
    document_id = message.get("document_id")
    data = message.get("data", {})
    
    # Add notification
    notification = {
        "id": message.get("message_id", str(uuid.uuid4())),
        "type": "document",
        "title": f"Document Update: {update_type}",
        "message": f"Document {document_id} was {update_type}",
        "timestamp": message.get("timestamp"),
        "data": data
    }
    st.session_state.notifications.append(notification)
    
    # Update UI if needed
    if update_type in ["version_created", "document_rollback"]:
        st.experimental_rerun()

def handle_user_update(message: Dict[str, Any]):
    """Handle user update messages."""
    update_type = message.get("update_type")
    user_id = message.get("user_id")
    data = message.get("data", {})
    
    # Add notification
    notification = {
        "id": message.get("message_id", str(uuid.uuid4())),
        "type": "user",
        "title": f"User Update: {update_type}",
        "message": f"User {user_id} {update_type}",
        "timestamp": message.get("timestamp"),
        "data": data
    }
    st.session_state.notifications.append(notification)

def handle_system_update(message: Dict[str, Any]):
    """Handle system update messages."""
    update_type = message.get("update_type")
    data = message.get("data", {})
    
    # Add notification
    notification = {
        "id": message.get("message_id", str(uuid.uuid4())),
        "type": "system",
        "title": f"System Update: {update_type}",
        "message": f"System {update_type}",
        "timestamp": message.get("timestamp"),
        "data": data
    }
    st.session_state.notifications.append(notification)
    
    # Update UI if needed
    if update_type in ["document_uploaded", "document_deleted"]:
        st.experimental_rerun()

def show_notifications():
    """Display notifications in the sidebar."""
    if not st.session_state.notifications:
        return
    
    with st.sidebar:
        st.subheader("Notifications")
        for notification in reversed(st.session_state.notifications[-5:]):  # Show last 5 notifications
            with st.expander(f"{notification['title']} ({notification['timestamp']})"):
                st.write(notification["message"])
                if notification["data"]:
                    st.json(notification["data"])

def subscribe_to_document(document_id: str):
    """Subscribe to document updates."""
    if document_id not in st.session_state.document_subscriptions:
        st.session_state.document_subscriptions.add(document_id)
        if st.session_state.ws_connected:
            # Send subscription message through WebSocket
            message_queue.put({
                "action": "subscribe_document",
                "document_id": document_id
            })

def unsubscribe_from_document(document_id: str):
    """Unsubscribe from document updates."""
    if document_id in st.session_state.document_subscriptions:
        st.session_state.document_subscriptions.remove(document_id)
        if st.session_state.ws_connected:
            # Send unsubscription message through WebSocket
            message_queue.put({
                "action": "unsubscribe_document",
                "document_id": document_id
            })

def subscribe_to_user(user_id: str):
    """Subscribe to user updates."""
    if user_id not in st.session_state.user_subscriptions:
        st.session_state.user_subscriptions.add(user_id)
        if st.session_state.ws_connected:
            # Send subscription message through WebSocket
            message_queue.put({
                "action": "subscribe_user",
                "user_id": user_id
            })

def unsubscribe_from_user(user_id: str):
    """Unsubscribe from user updates."""
    if user_id in st.session_state.user_subscriptions:
        st.session_state.user_subscriptions.remove(user_id)
        if st.session_state.ws_connected:
            # Send unsubscription message through WebSocket
            message_queue.put({
                "action": "unsubscribe_user",
                "user_id": user_id
            })

def login(email: str, password: str) -> bool:
    """Login user and initialize WebSocket connection."""
    try:
        response = requests.post(
            f"{API_URL}/auth/login",
            data={"username": email, "password": password}
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.token = data["access_token"]
            st.session_state.user = data["user"]
            
            # Start WebSocket connection
            if not st.session_state.ws_connected:
                threading.Thread(target=start_websocket_thread, daemon=True).start()
            
            return True
        return False
    except Exception as e:
        st.error(f"Login error: {str(e)}")
        return False

def logout():
    """Logout user and close WebSocket connection."""
    st.session_state.token = None
    st.session_state.user = None
    st.session_state.ws_connected = False
    st.session_state.notifications = []
    st.session_state.document_subscriptions = set()
    st.session_state.user_subscriptions = set()

def register(email: str, password: str, full_name: str) -> bool:
    """Register a new user."""
    try:
        response = requests.post(
            f"{API_URL}/auth/register",
            json={
                "email": email,
                "password": password,
                "full_name": full_name
            }
        )
        return response.status_code == 200
    except Exception as e:
        st.error(f"Registration error: {str(e)}")
        return False

def show_auth_ui():
    """Show authentication UI."""
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if login(email, password):
                    st.success("Login successful!")
                    st.experimental_rerun()
                else:
                    st.error("Invalid credentials")
    
    with tab2:
        with st.form("register_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            full_name = st.text_input("Full Name")
            submit = st.form_submit_button("Register")
            
            if submit:
                if password != confirm_password:
                    st.error("Passwords do not match")
                elif register(email, password, full_name):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Registration failed")

def show_document_versions(document_id: str):
    """Show document versions and versioning controls."""
    try:
        response = requests.get(
            f"{API_URL}/documents/{document_id}/versions",
            headers=get_auth_headers()
        )
        if response.status_code == 200:
            versions = response.json()
            
            st.subheader("Document Versions")
            for version in versions:
                with st.expander(f"Version {version['version_number']} ({version['created_at']})"):
                    st.write(f"Created by: {version['created_by']}")
                    if version.get("changes"):
                        st.write("Changes:", version["changes"])
                    
                    # Version actions
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("View", key=f"view_{version['version_id']}"):
                            st.session_state["viewing_version"] = version
                    with col2:
                        if st.button("Rollback", key=f"rollback_{version['version_id']}"):
                            if st.session_state.user["role"] in ["ADMIN", "EDITOR"]:
                                response = requests.post(
                                    f"{API_URL}/documents/{document_id}/versions/{version['version_id']}/rollback",
                                    headers=get_auth_headers()
                                )
                                if response.status_code == 200:
                                    st.success("Document rolled back successfully")
                                    st.experimental_rerun()
                                else:
                                    st.error("Rollback failed")
                            else:
                                st.error("Only admins and editors can rollback documents")
            
            # Version comparison
            st.subheader("Compare Versions")
            col1, col2 = st.columns(2)
            with col1:
                version1 = st.selectbox(
                    "Select first version",
                    options=[v["version_id"] for v in versions],
                    format_func=lambda x: f"Version {next(v['version_number'] for v in versions if v['version_id'] == x)}"
                )
            with col2:
                version2 = st.selectbox(
                    "Select second version",
                    options=[v["version_id"] for v in versions],
                    format_func=lambda x: f"Version {next(v['version_number'] for v in versions if v['version_id'] == x)}"
                )
            
            if st.button("Compare Versions"):
                response = requests.get(
                    f"{API_URL}/documents/{document_id}/versions/compare",
                    params={"version_id1": version1, "version_id2": version2},
                    headers=get_auth_headers()
                )
                if response.status_code == 200:
                    comparison = response.json()
                    st.json(comparison)
                else:
                    st.error("Failed to compare versions")
            
            # Add version tag
            st.subheader("Add Version Tag")
            with st.form("add_tag_form"):
                version_id = st.selectbox(
                    "Select version",
                    options=[v["version_id"] for v in versions],
                    format_func=lambda x: f"Version {next(v['version_number'] for v in versions if v['version_id'] == x)}"
                )
                tag_name = st.text_input("Tag Name")
                description = st.text_area("Description")
                submit = st.form_submit_button("Add Tag")
                
                if submit:
                    response = requests.post(
                        f"{API_URL}/documents/{document_id}/versions/{version_id}/tags",
                        params={"tag_name": tag_name, "description": description},
                        headers=get_auth_headers()
                    )
                    if response.status_code == 200:
                        st.success("Tag added successfully")
                    else:
                        st.error("Failed to add tag")
        
    except Exception as e:
        st.error(f"Error loading document versions: {str(e)}")

def main():
    """Main application function."""
    st.set_page_config(
        page_title="Document Management System",
        page_icon="📚",
        layout="wide"
    )
    
    # Process WebSocket messages
    process_websocket_messages()
    
    # Show notifications
    show_notifications()
    
    # Check authentication
    if not st.session_state.token:
        show_auth_ui()
        return
    
    # Sidebar
    with st.sidebar:
        st.write(f"Logged in as: {st.session_state.user['email']}")
        if st.button("Logout"):
            logout()
            st.experimental_rerun()
    
    # Main content
    st.title("Document Management System")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Documents", "Query", "Analytics", "User Management"]
    )
    
    if page == "Documents":
        show_documents_page()
    elif page == "Query":
        show_query_page()
    elif page == "Analytics":
        show_analytics_page()
    elif page == "User Management":
        if st.session_state.user["role"] == "ADMIN":
            show_user_management_page()
        else:
            st.error("Access denied. Admin privileges required.")

def show_documents_page():
    """Show documents page with versioning support."""
    st.header("Documents")
    
    # Upload document
    with st.expander("Upload Document"):
        with st.form("upload_form"):
            title = st.text_input("Title")
            content = st.text_area("Content")
            metadata = st.text_area("Metadata (JSON)")
            submit = st.form_submit_button("Upload")
            
            if submit:
                try:
                    metadata_dict = json.loads(metadata) if metadata else {}
                    response = requests.post(
                        f"{API_URL}/documents/upload",
                        json={
                            "title": title,
                            "content": content,
                            "metadata": metadata_dict
                        },
                        headers=get_auth_headers()
                    )
                    if response.status_code == 200:
                        st.success("Document uploaded successfully")
                        st.experimental_rerun()
                    else:
                        st.error("Upload failed")
                except json.JSONDecodeError:
                    st.error("Invalid JSON metadata")
                except Exception as e:
                    st.error(f"Upload error: {str(e)}")
    
    # List documents
    try:
        response = requests.get(
            f"{API_URL}/documents",
            headers=get_auth_headers()
        )
        if response.status_code == 200:
            documents = response.json()
            
            for doc in documents:
                with st.expander(f"{doc['title']} ({doc['created_at']})"):
                    st.write(f"ID: {doc['id']}")
                    st.write(f"Created by: {doc['created_by']}")
                    
                    # Document actions
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("View", key=f"view_{doc['id']}"):
                            st.session_state["viewing_document"] = doc
                    with col2:
                        if st.button("Versions", key=f"versions_{doc['id']}"):
                            st.session_state["viewing_versions"] = doc["id"]
                    with col3:
                        if st.button("Delete", key=f"delete_{doc['id']}"):
                            if st.session_state.user["role"] in ["ADMIN", "EDITOR"]:
                                response = requests.delete(
                                    f"{API_URL}/documents/{doc['id']}",
                                    headers=get_auth_headers()
                                )
                                if response.status_code == 200:
                                    st.success("Document deleted successfully")
                                    st.experimental_rerun()
                                else:
                                    st.error("Deletion failed")
                            else:
                                st.error("Only admins and editors can delete documents")
                    
                    # Subscribe to updates
                    if st.button("Subscribe to Updates", key=f"subscribe_{doc['id']}"):
                        subscribe_to_document(doc["id"])
                        st.success(f"Subscribed to updates for {doc['title']}")
            
            # Show document viewer
            if "viewing_document" in st.session_state:
                doc = st.session_state["viewing_document"]
                st.subheader("Document Viewer")
                st.write(f"Title: {doc['title']}")
                st.write("Content:")
                st.text_area("", doc["content"], height=300)
                st.write("Metadata:")
                st.json(doc["metadata"])
            
            # Show versioning interface
            if "viewing_versions" in st.session_state:
                show_document_versions(st.session_state["viewing_versions"])
        
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")

def show_query_page():
    """Show query page with real-time updates."""
    st.header("Query Documents")
    
    # Query form
    with st.form("query_form"):
        query = st.text_input("Enter your query")
        document_id = st.selectbox(
            "Select document (optional)",
            options=[""] + [doc["id"] for doc in st.session_state.get("documents", [])]
        )
        submit = st.form_submit_button("Submit Query")
        
        if submit:
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={
                        "query": query,
                        "document_id": document_id if document_id else None
                    },
                    headers=get_auth_headers()
                )
                if response.status_code == 200:
                    result = response.json()
                    st.subheader("Query Result")
                    st.write(result["response"])
                    
                    # Feedback form
                    with st.form("feedback_form"):
                        rating = st.slider("Rate this response", 1, 5)
                        feedback = st.text_area("Additional feedback")
                        submit_feedback = st.form_submit_button("Submit Feedback")
                        
                        if submit_feedback:
                            feedback_response = requests.post(
                                f"{API_URL}/feedback",
                                json={
                                    "document_id": document_id,
                                    "query": query,
                                    "response": result["response"],
                                    "rating": rating,
                                    "feedback": feedback
                                },
                                headers=get_auth_headers()
                            )
                            if feedback_response.status_code == 200:
                                st.success("Feedback submitted successfully")
                            else:
                                st.error("Failed to submit feedback")
                else:
                    st.error("Query failed")
            except Exception as e:
                st.error(f"Query error: {str(e)}")

def show_analytics_page():
    """Show analytics page with caching support."""
    st.header("Analytics")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date")
    with col2:
        end_date = st.date_input("End Date")
    
    # Document selector
    document_id = st.selectbox(
        "Select document (optional)",
        options=[""] + [doc["id"] for doc in st.session_state.get("documents", [])]
    )
    
    if st.button("Load Analytics"):
        try:
            response = requests.get(
                f"{API_URL}/analytics/feedback",
                params={
                    "document_id": document_id if document_id else None,
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None
                },
                headers=get_auth_headers()
            )
            if response.status_code == 200:
                analytics = response.json()
                
                # Display analytics
                st.subheader("Query Performance")
                st.write(f"Total Queries: {analytics['query_metrics']['total_queries']}")
                st.write(f"Average Response Time: {analytics['query_metrics']['avg_response_time']:.2f}s")
                
                # Plot query success rate
                fig = go.Figure(data=[
                    go.Bar(
                        x=["Success Rate"],
                        y=[analytics["query_metrics"]["success_rate"]],
                        name="Success Rate"
                    )
                ])
                fig.update_layout(title="Query Success Rate")
                st.plotly_chart(fig)
                
                # Plot feedback distribution
                feedback_data = pd.DataFrame(analytics["feedback_metrics"]["rating_distribution"])
                fig = px.pie(
                    feedback_data,
                    values="count",
                    names="rating",
                    title="Feedback Distribution"
                )
                st.plotly_chart(fig)
                
                # System health
                st.subheader("System Health")
                health_data = pd.DataFrame(analytics["system_health"])
                st.line_chart(health_data.set_index("timestamp"))
            else:
                st.error("Failed to load analytics")
        except Exception as e:
            st.error(f"Analytics error: {str(e)}")

def show_user_management_page():
    """Show user management page with real-time updates."""
    st.header("User Management")
    
    # List users
    try:
        response = requests.get(
            f"{API_URL}/auth/users",
            headers=get_auth_headers()
        )
        if response.status_code == 200:
            users = response.json()
            
            for user in users:
                with st.expander(f"{user['email']} ({user['role']})"):
                    st.write(f"ID: {user['id']}")
                    st.write(f"Full Name: {user['full_name']}")
                    st.write(f"Created at: {user['created_at']}")
                    
                    # User actions
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Update Role", key=f"role_{user['id']}"):
                            new_role = st.selectbox(
                                "Select new role",
                                options=["ADMIN", "EDITOR", "VIEWER"],
                                key=f"role_select_{user['id']}"
                            )
                            if st.button("Confirm", key=f"confirm_role_{user['id']}"):
                                response = requests.put(
                                    f"{API_URL}/auth/users/{user['id']}",
                                    json={"role": new_role},
                                    headers=get_auth_headers()
                                )
                                if response.status_code == 200:
                                    st.success("Role updated successfully")
                                    st.experimental_rerun()
                                else:
                                    st.error("Failed to update role")
                    with col2:
                        if st.button("Subscribe to Updates", key=f"subscribe_user_{user['id']}"):
                            subscribe_to_user(user["id"])
                            st.success(f"Subscribed to updates for {user['email']}")
            
    except Exception as e:
        st.error(f"Error loading users: {str(e)}")

if __name__ == "__main__":
    main() 