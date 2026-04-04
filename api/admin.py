"""
Admin Panel - Authentication and Settings Management
Supports environment variable overrides for persistent configuration.
"""
import json
import hashlib
import secrets
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict
from pydantic import BaseModel
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SETTINGS_FILE = Path(__file__).parent.parent / "admin_settings.json"
SESSIONS_FILE = Path("/tmp/admin_sessions.json")  # Use /tmp for persistence within instance
RUNTIME_SETTINGS_FILE = Path("/tmp/admin_settings_runtime.json")  # Runtime overrides in /tmp

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

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def load_settings() -> Dict:
    """Load settings with priority: /tmp runtime file > env vars > repo file > defaults"""
    
    # Start with defaults
    settings = {
        "llm_provider": "openrouter",
        "llm_api_url": "https://openrouter.ai/api/v1",
        "llm_api_key": "",
        "llm_model": "google/gemini-2.0-flash-001",
        "admin_username": "admin",
        "admin_password_hash": hash_password("admin123")
    }
    
    # Load from repo file (lowest priority)
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE) as f:
                file_settings = json.load(f)
                settings.update(file_settings)
        except Exception as e:
            print(f"Error loading settings file: {e}")
    
    # Override with environment variables (medium priority)
    env_overrides = {
        "llm_provider": os.environ.get("LLM_PROVIDER"),
        "llm_api_url": os.environ.get("LLM_API_URL"),
        "llm_api_key": os.environ.get("LLM_API_KEY"),
        "llm_model": os.environ.get("LLM_MODEL"),
        "admin_username": os.environ.get("ADMIN_USERNAME"),
        "admin_password_hash": os.environ.get("ADMIN_PASSWORD_HASH"),
    }
    for k, v in env_overrides.items():
        if v:  # Only override if env var is set
            settings[k] = v
    
    # Override with /tmp runtime settings (highest priority - set via admin panel)
    if RUNTIME_SETTINGS_FILE.exists():
        try:
            with open(RUNTIME_SETTINGS_FILE) as f:
                runtime_settings = json.load(f)
                settings.update(runtime_settings)
        except Exception as e:
            print(f"Error loading runtime settings: {e}")
    
    return settings

def save_settings(settings: Dict):
    """Save settings to BOTH the repo file AND the /tmp runtime file for persistence"""
    # Save to /tmp first (persists within the running container instance)
    try:
        with open(RUNTIME_SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        print(f"Settings saved to {RUNTIME_SETTINGS_FILE}")
    except Exception as e:
        print(f"Error saving to /tmp: {e}")
    
    # Also save to the repo file
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        print(f"Settings saved to {SETTINGS_FILE}")
    except Exception as e:
        print(f"Error saving to repo file: {e}")

def load_sessions() -> Dict:
    if SESSIONS_FILE.exists():
        try:
            with open(SESSIONS_FILE) as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_sessions(sessions: Dict):
    try:
        with open(SESSIONS_FILE, 'w') as f:
            json.dump(sessions, f)
    except Exception as e:
        print(f"Error saving sessions: {e}")

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
        return {
            "llm_provider": settings.get("llm_provider", ""),
            "llm_api_url": settings.get("llm_api_url", ""),
            "llm_api_key": settings.get("llm_api_key", ""),
            "llm_model": settings.get("llm_model", "")
        }

    @app.post("/api/admin/settings")
    async def update_settings(
        new_settings: LLMSettings,
        username: str = Depends(get_current_admin)
    ):
        settings = load_settings()
        settings["llm_provider"] = new_settings.llm_provider
        settings["llm_api_url"] = new_settings.llm_api_url.rstrip("/")
        settings["llm_api_key"] = new_settings.llm_api_key.strip()
        settings["llm_model"] = new_settings.llm_model.strip()
        save_settings(settings)
        
        # Verify saved
        verify = load_settings()
        saved_key = verify.get("llm_api_key", "")
        print(f"Settings saved. API key present: {bool(saved_key)}, length: {len(saved_key)}")
        
        return {
            "message": "Settings updated successfully",
            "api_key_saved": bool(saved_key),
            "model": verify.get("llm_model", "")
        }

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

        kg_stats = {}
        kg_file = project_dir / "knowledge_graph" / "graph_statistics.json"
        if kg_file.exists():
            with open(kg_file) as f:
                kg_stats = json.load(f)

        return {
            "papers_processed": 317,
            "chunks_created": 9003,
            "embeddings": 6469,
            "entities": kg_stats.get("total_entities", 155),
            "relationships": kg_stats.get("total_relationships", 1014),
            "contradictions": kg_stats.get("total_contradictions", 87)
        }

    return app
