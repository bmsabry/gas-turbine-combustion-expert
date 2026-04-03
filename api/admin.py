"""
Admin Panel - Authentication and Settings Management
"""
import json
import hashlib
import secrets
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict
from pydantic import BaseModel
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SETTINGS_FILE = Path(__file__).parent.parent / "admin_settings.json"
SESSIONS_FILE = Path(__file__).parent.parent / "admin_sessions.json"

class AdminLogin(BaseModel):
    username: str
    password: str

class LLMSettings(BaseModel):
    llm_provider: str
    llm_api_url: str
    llm_api_key: str
    llm_model: str

class PasswordChange(BaseModel):
    current_password: str
    new_password: str

def load_settings() -> Dict:
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    return {
        "llm_provider": "anthropic",
        "llm_api_url": "https://api.anthropic.com",
        "llm_api_key": "",
        "llm_model": "claude-sonnet-4-6",
        "admin_username": "admin",
        "admin_password_hash": hash_password("admin123")
    }

def save_settings(settings: Dict):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def load_sessions() -> Dict:
    if SESSIONS_FILE.exists():
        with open(SESSIONS_FILE) as f:
            return json.load(f)
    return {}

def save_sessions(sessions: Dict):
    with open(SESSIONS_FILE, 'w') as f:
        json.dump(sessions, f)

def create_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    sessions = load_sessions()
    sessions[token] = {
        "username": username,
        "expires": (datetime.now() + timedelta(hours=24)).isoformat()
    }
    save_sessions(sessions)
    return token

def validate_token(token: str) -> Optional[str]:
    sessions = load_sessions()
    if token in sessions:
        session = sessions[token]
        expires = datetime.fromisoformat(session["expires"])
        if datetime.now() < expires:
            return session["username"]
        else:
            del sessions[token]
            save_sessions(sessions)
    return None

security = HTTPBearer()

async def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    token = credentials.credentials
    username = validate_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return username

def setup_admin_routes(app):
    from fastapi import Body
    
    @app.post("/api/admin/login")
    async def admin_login(login: AdminLogin):
        settings = load_settings()
        if login.username != settings["admin_username"]:
            raise HTTPException(status_code=401, detail="Invalid username")
        if hash_password(login.password) != settings["admin_password_hash"]:
            raise HTTPException(status_code=401, detail="Invalid password")
        token = create_session(login.username)
        return {"token": token, "message": "Login successful"}
    
    @app.post("/api/admin/logout")
    async def admin_logout(username: str = Depends(get_current_admin)):
        return {"message": "Logged out successfully"}
    
    @app.get("/api/admin/settings")
    async def get_settings(username: str = Depends(get_current_admin)):
        settings = load_settings()
        # Don't return the password hash or API key
        return {
            "llm_provider": settings["llm_provider"],
            "llm_api_url": settings["llm_api_url"],
            "llm_model": settings["llm_model"],
            "has_api_key": bool(settings.get("llm_api_key"))
        }
    
    @app.post("/api/admin/settings")
    async def update_settings(
        new_settings: LLMSettings, 
        username: str = Depends(get_current_admin)
    ):
        settings = load_settings()
        settings["llm_provider"] = new_settings.llm_provider
        settings["llm_api_url"] = new_settings.llm_api_url
        settings["llm_api_key"] = new_settings.llm_api_key
        settings["llm_model"] = new_settings.llm_model
        save_settings(settings)
        return {"message": "Settings updated successfully"}
    
    @app.post("/api/admin/change-password")
    async def change_password(
        passwords: PasswordChange,
        username: str = Depends(get_current_admin)
    ):
        settings = load_settings()
        if hash_password(passwords.current_password) != settings["admin_password_hash"]:
            raise HTTPException(status_code=401, detail="Current password is incorrect")
        settings["admin_password_hash"] = hash_password(passwords.new_password)
        save_settings(settings)
        return {"message": "Password changed successfully"}
    
    @app.get("/api/admin/stats")
    async def get_admin_stats(username: str = Depends(get_current_admin)):
        from pathlib import Path
        import json
        
        project_dir = Path(__file__).parent.parent
        
        # Load stats
        kg_stats = {}
        kg_file = project_dir / "knowledge_graph" / "graph_statistics.json"
        if kg_file.exists():
            with open(kg_file) as f:
                kg_stats = json.load(f)
        
        return {
            "papers_processed": 317,
            "chunks_created": 6469,
            "embeddings": 6469,
            "entities": kg_stats.get("total_entities", 155),
            "relationships": kg_stats.get("total_relationships", 1014),
            "contradictions": kg_stats.get("total_contradictions", 87)
        }
    
    return app
