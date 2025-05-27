import logging
import json
import asyncio
from typing import Dict, Set, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections and broadcasts."""
    
    def __init__(self):
        """Initialize the connection manager."""
        # Active connections by user_id
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Document subscriptions by document_id
        self.document_subscriptions: Dict[str, Set[str]] = {}
        # User subscriptions by user_id
        self.user_subscriptions: Dict[str, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str) -> None:
        """Connect a new WebSocket client."""
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)
        logger.info(f"User {user_id} connected. Active connections: {len(self.active_connections[user_id])}")
    
    def disconnect(self, websocket: WebSocket, user_id: str) -> None:
        """Disconnect a WebSocket client."""
        if user_id in self.active_connections:
            self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        
        # Clean up subscriptions
        for doc_id, subscribers in self.document_subscriptions.items():
            if user_id in subscribers:
                subscribers.remove(user_id)
                if not subscribers:
                    del self.document_subscriptions[doc_id]
        
        if user_id in self.user_subscriptions:
            del self.user_subscriptions[user_id]
        
        logger.info(f"User {user_id} disconnected")
    
    async def subscribe_to_document(self, user_id: str, document_id: str) -> None:
        """Subscribe a user to document updates."""
        if document_id not in self.document_subscriptions:
            self.document_subscriptions[document_id] = set()
        self.document_subscriptions[document_id].add(user_id)
        logger.info(f"User {user_id} subscribed to document {document_id}")
    
    async def unsubscribe_from_document(self, user_id: str, document_id: str) -> None:
        """Unsubscribe a user from document updates."""
        if document_id in self.document_subscriptions:
            self.document_subscriptions[document_id].discard(user_id)
            if not self.document_subscriptions[document_id]:
                del self.document_subscriptions[document_id]
        logger.info(f"User {user_id} unsubscribed from document {document_id}")
    
    async def subscribe_to_user(self, user_id: str, target_user_id: str) -> None:
        """Subscribe a user to another user's updates."""
        if user_id not in self.user_subscriptions:
            self.user_subscriptions[user_id] = set()
        self.user_subscriptions[user_id].add(target_user_id)
        logger.info(f"User {user_id} subscribed to user {target_user_id}")
    
    async def unsubscribe_from_user(self, user_id: str, target_user_id: str) -> None:
        """Unsubscribe a user from another user's updates."""
        if user_id in self.user_subscriptions:
            self.user_subscriptions[user_id].discard(target_user_id)
            if not self.user_subscriptions[user_id]:
                del self.user_subscriptions[user_id]
        logger.info(f"User {user_id} unsubscribed from user {target_user_id}")
    
    async def broadcast_to_user(
        self,
        user_id: str,
        message: Dict[str, Any],
        exclude_sender: Optional[str] = None
    ) -> None:
        """Broadcast a message to all connections of a specific user."""
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json(message)
                except WebSocketDisconnect:
                    self.disconnect(connection, user_id)
                except Exception as e:
                    logger.error(f"Error broadcasting to user {user_id}: {str(e)}")
    
    async def broadcast_to_document(
        self,
        document_id: str,
        message: Dict[str, Any],
        exclude_sender: Optional[str] = None
    ) -> None:
        """Broadcast a message to all subscribers of a document."""
        if document_id in self.document_subscriptions:
            for user_id in self.document_subscriptions[document_id]:
                if user_id != exclude_sender:
                    await self.broadcast_to_user(user_id, message)
    
    async def broadcast_to_subscribers(
        self,
        user_id: str,
        message: Dict[str, Any],
        exclude_sender: Optional[str] = None
    ) -> None:
        """Broadcast a message to all subscribers of a user."""
        if user_id in self.user_subscriptions:
            for subscriber_id in self.user_subscriptions[user_id]:
                if subscriber_id != exclude_sender:
                    await self.broadcast_to_user(subscriber_id, message)

class WebSocketService:
    """Service for handling WebSocket connections and real-time updates."""
    
    def __init__(self):
        """Initialize the WebSocket service."""
        self.manager = ConnectionManager()
    
    async def handle_connection(
        self,
        websocket: WebSocket,
        user_id: str
    ) -> None:
        """Handle a new WebSocket connection."""
        await self.manager.connect(websocket, user_id)
        try:
            while True:
                data = await websocket.receive_json()
                await self._handle_message(websocket, user_id, data)
        except WebSocketDisconnect:
            self.manager.disconnect(websocket, user_id)
        except Exception as e:
            logger.error(f"Error handling WebSocket connection: {str(e)}")
            self.manager.disconnect(websocket, user_id)
    
    async def _handle_message(
        self,
        websocket: WebSocket,
        user_id: str,
        data: Dict[str, Any]
    ) -> None:
        """Handle incoming WebSocket messages."""
        try:
            action = data.get("action")
            if not action:
                raise ValueError("No action specified")
            
            if action == "subscribe_document":
                document_id = data.get("document_id")
                if not document_id:
                    raise ValueError("No document_id specified")
                await self.manager.subscribe_to_document(user_id, document_id)
                await websocket.send_json({
                    "type": "subscription_confirmed",
                    "action": "subscribe_document",
                    "document_id": document_id
                })
            
            elif action == "unsubscribe_document":
                document_id = data.get("document_id")
                if not document_id:
                    raise ValueError("No document_id specified")
                await self.manager.unsubscribe_from_document(user_id, document_id)
                await websocket.send_json({
                    "type": "subscription_confirmed",
                    "action": "unsubscribe_document",
                    "document_id": document_id
                })
            
            elif action == "subscribe_user":
                target_user_id = data.get("user_id")
                if not target_user_id:
                    raise ValueError("No user_id specified")
                await self.manager.subscribe_to_user(user_id, target_user_id)
                await websocket.send_json({
                    "type": "subscription_confirmed",
                    "action": "subscribe_user",
                    "user_id": target_user_id
                })
            
            elif action == "unsubscribe_user":
                target_user_id = data.get("user_id")
                if not target_user_id:
                    raise ValueError("No user_id specified")
                await self.manager.unsubscribe_from_user(user_id, target_user_id)
                await websocket.send_json({
                    "type": "subscription_confirmed",
                    "action": "unsubscribe_user",
                    "user_id": target_user_id
                })
            
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
    
    async def notify_document_update(
        self,
        document_id: str,
        update_type: str,
        data: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> None:
        """Notify subscribers about a document update."""
        message = {
            "type": "document_update",
            "document_id": document_id,
            "update_type": update_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "message_id": str(uuid.uuid4())
        }
        await self.manager.broadcast_to_document(document_id, message, exclude_sender=user_id)
    
    async def notify_user_update(
        self,
        user_id: str,
        update_type: str,
        data: Dict[str, Any]
    ) -> None:
        """Notify subscribers about a user update."""
        message = {
            "type": "user_update",
            "user_id": user_id,
            "update_type": update_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "message_id": str(uuid.uuid4())
        }
        await self.manager.broadcast_to_subscribers(user_id, message)
    
    async def notify_system_update(
        self,
        update_type: str,
        data: Dict[str, Any],
        target_users: Optional[Set[str]] = None
    ) -> None:
        """Notify users about a system update."""
        message = {
            "type": "system_update",
            "update_type": update_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "message_id": str(uuid.uuid4())
        }
        
        if target_users:
            for user_id in target_users:
                await self.manager.broadcast_to_user(user_id, message)
        else:
            # Broadcast to all connected users
            for user_id in self.manager.active_connections:
                await self.manager.broadcast_to_user(user_id, message) 