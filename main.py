import os
import subprocess
import mimetypes
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import json
import hashlib

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import jwt

# ── Config ────────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production-please")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24
MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", "./media"))
THUMBNAIL_DIR = Path(os.getenv("THUMBNAIL_DIR", "./thumbnails"))
PORT = int(os.getenv("PORT", 8000))

MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_VIDEO = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".flv", ".wmv"}
SUPPORTED_AUDIO = {".mp3", ".flac", ".wav", ".aac", ".ogg", ".m4a", ".opus"}
SUPPORTED_IMAGE = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="StreamVault", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login", auto_error=False)

# ── Users (JSON file store) ───────────────────────────────────────────────────
USERS_FILE = Path("users.json")

def load_users():
    if not USERS_FILE.exists():
        default = {
            "admin": {
                "username": "admin",
                # Default password: "admin" — prompt user to change this!
                "password_hash": hashlib.sha256(b"admin").hexdigest(),
                "role": "admin",
                "created_at": datetime.utcnow().isoformat(),
            }
        }
        USERS_FILE.write_text(json.dumps(default, indent=2))
    return json.loads(USERS_FILE.read_text())

def save_users(users):
    USERS_FILE.write_text(json.dumps(users, indent=2))

def hash_pw(p: str) -> str:
    return hashlib.sha256(p.encode()).hexdigest()

# ── Auth helpers ──────────────────────────────────────────────────────────────
def create_token(data: dict, expires: timedelta = None) -> str:
    d = data.copy()
    d["exp"] = datetime.utcnow() + (expires or timedelta(minutes=15))
    return jwt.encode(d, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
    users = load_users()
    if username not in users:
        raise HTTPException(status_code=401, detail="User not found")
    return users[username]

# ── Models ────────────────────────────────────────────────────────────────────
class UserCreate(BaseModel):
    username: str
    password: str

class MediaItem(BaseModel):
    id: str
    name: str
    path: str
    type: str
    size: int
    modified: str

# ── Media scanning ────────────────────────────────────────────────────────────
def scan_media() -> list[MediaItem]:
    items = []
    for path in sorted(MEDIA_ROOT.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in SUPPORTED_VIDEO:
            mt = "video"
        elif ext in SUPPORTED_AUDIO:
            mt = "audio"
        elif ext in SUPPORTED_IMAGE:
            mt = "image"
        else:
            continue
        rel = path.relative_to(MEDIA_ROOT)
        items.append(MediaItem(
            id=hashlib.md5(str(rel).encode()).hexdigest(),
            name=path.stem,
            path=str(rel),
            type=mt,
            size=path.stat().st_size,
            modified=datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        ))
    return items

def get_by_id(mid: str) -> Optional[MediaItem]:
    for i in scan_media():
        if i.id == mid:
            return i
    return None

# ── HTML serving ──────────────────────────────────────────────────────────────
INDEX_HTML = Path("index.html")

@app.get("/", response_class=HTMLResponse)
async def index():
    if not INDEX_HTML.exists():
        return HTMLResponse("<h1>StreamVault</h1><p>index.html not found next to main.py</p>", status_code=500)
    return HTMLResponse(INDEX_HTML.read_text())

# ── Auth endpoints ────────────────────────────────────────────────────────────
@app.post("/api/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    users = load_users()
    user = users.get(form_data.username)
    if not user or user["password_hash"] != hash_pw(form_data.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(
        {"sub": user["username"]},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": user["username"],
        "role": user["role"],
        # FIX: frontend checks is_admin to show the Admin nav item
        "is_admin": user["role"] == "admin",
    }

@app.post("/api/auth/register")
async def register(user_data: UserCreate):
    users = load_users()
    if user_data.username in users:
        raise HTTPException(status_code=400, detail="Username already exists")
    users[user_data.username] = {
        "username": user_data.username,
        "password_hash": hash_pw(user_data.password),
        "role": "viewer",
        "created_at": datetime.utcnow().isoformat(),
    }
    save_users(users)
    token = create_token(
        {"sub": user_data.username},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": user_data.username,
        "role": "viewer",
        "is_admin": False,
    }

@app.get("/api/auth/me")
async def me(user=Depends(get_current_user)):
    return {
        "username": user["username"],
        "role": user["role"],
        # FIX: frontend checks is_admin here too (on page refresh)
        "is_admin": user["role"] == "admin",
    }

# ── Library endpoints ─────────────────────────────────────────────────────────
@app.get("/api/library")
async def get_library(
    type: Optional[str] = None,
    search: Optional[str] = None,
    user=Depends(get_current_user),
):
    items = scan_media()
    if type:
        items = [i for i in items if i.type == type]
    if search:
        items = [i for i in items if search.lower() in i.name.lower()]
    return items

@app.get("/api/library/stats")
async def get_stats(user=Depends(get_current_user)):
    items = scan_media()
    return {
        "total_files": len(items),
        "total_size": sum(i.size for i in items),
        "video_count": sum(1 for i in items if i.type == "video"),
        "audio_count": sum(1 for i in items if i.type == "audio"),
        "image_count": sum(1 for i in items if i.type == "image"),
    }

# ── Streaming ─────────────────────────────────────────────────────────────────
@app.get("/api/stream/{media_id}")
async def stream_media(
    media_id: str,
    request: Request,
    transcode: bool = False,
    user=Depends(get_current_user),
):
    item = get_by_id(media_id)
    if not item:
        raise HTTPException(status_code=404, detail="Media not found")
    file_path = MEDIA_ROOT / item.path
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")

    # Transcode via FFmpeg
    if transcode and item.type == "video":
        cmd = [
            "ffmpeg", "-i", str(file_path),
            "-vcodec", "libx264", "-preset", "ultrafast",
            "-acodec", "aac",
            "-f", "mp4", "-movflags", "frag_keyframe+empty_moov",
            "pipe:1",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        def gen():
            try:
                while chunk := proc.stdout.read(65536):
                    yield chunk
            finally:
                proc.kill()
        return StreamingResponse(gen(), media_type="video/mp4")

    # Direct stream with Range support
    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")
    mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"

    if range_header:
        start_str, end_str = range_header.replace("bytes=", "").split("-")
        start = int(start_str)
        end = int(end_str) if end_str else file_size - 1
        chunk_size = end - start + 1

        def iter_file():
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = chunk_size
                while remaining:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            iter_file(),
            status_code=206,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size),
                "Content-Type": mime_type,
            },
        )

    def iter_file():
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                yield chunk

    return StreamingResponse(
        iter_file(),
        headers={
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
            "Content-Type": mime_type,
        },
    )

# ── Thumbnails ────────────────────────────────────────────────────────────────
@app.get("/api/thumbnail/{media_id}")
async def get_thumbnail(media_id: str, user=Depends(get_current_user)):
    item = get_by_id(media_id)
    if not item or item.type != "video":
        raise HTTPException(status_code=404, detail="Not found")
    file_path = MEDIA_ROOT / item.path
    thumb_path = THUMBNAIL_DIR / f"{media_id}.jpg"
    if not thumb_path.exists():
        r = subprocess.run(
            ["ffmpeg", "-i", str(file_path), "-ss", "00:00:05",
             "-vframes", "1", "-vf", "scale=320:-1", str(thumb_path)],
            capture_output=True,
        )
        if r.returncode != 0:
            raise HTTPException(status_code=500, detail="Thumbnail generation failed")
    return FileResponse(str(thumb_path), media_type="image/jpeg")

# ── Admin endpoints ───────────────────────────────────────────────────────────
@app.get("/api/admin/users")
async def list_users(user=Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    users = load_users()
    return [
        {"username": u["username"], "role": u["role"], "created_at": u["created_at"]}
        for u in users.values()
    ]

@app.delete("/api/admin/users/{username}")
async def delete_user(username: str, user=Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    if username == user["username"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    users = load_users()
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    del users[username]
    save_users(users)
    return {"message": "User deleted"}

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
