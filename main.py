"""
StreamVault — Full Backend
Matches every API endpoint called by index.html
"""
import os
import subprocess
import mimetypes
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import json
import hashlib

from fastapi import FastAPI, HTTPException, Depends, Request, Query
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import jwt

# ── Config ─────────────────────────────────────────────────────────────────
SECRET_KEY   = os.getenv("SECRET_KEY", "change-me-in-production-please")
ALGORITHM    = "HS256"
TOKEN_EXPIRE = 60 * 24          # minutes
MEDIA_ROOT   = Path(os.getenv("MEDIA_ROOT",   "./media"))
THUMB_DIR    = Path(os.getenv("THUMBNAIL_DIR","./thumbnails"))
PORT         = int(os.getenv("PORT", 8000))

MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True,  exist_ok=True)

SUPPORTED_VIDEO = {".mp4",".mkv",".avi",".mov",".webm",".m4v",".flv",".wmv"}
SUPPORTED_AUDIO = {".mp3",".flac",".wav",".aac",".ogg",".m4a",".opus"}
SUPPORTED_IMAGE = {".jpg",".jpeg",".png",".gif",".webp"}

# ── Simple JSON "database" ─────────────────────────────────────────────────
DATA_FILE = Path("data.json")

def _default_data():
    return {
        "users": {
            "admin": {
                "username":      "admin",
                "password_hash": hashlib.sha256(b"admin").hexdigest(),
                "is_admin":      True,
                "created_at":    datetime.utcnow().isoformat(),
            }
        },
        "libraries":      {},   # lib_id  -> library dict
        "media":          {},   # media_id -> media dict
        "progress":       {},   # "user:media_id" -> progress dict
        "playlists":      {},   # playlist_id -> playlist dict
        "playlist_items": {},   # playlist_id -> [media_id, ...]
        "_seq": {"lib": 0, "media": 0, "playlist": 0},
    }

_db_lock = threading.Lock()

def load_db() -> dict:
    if not DATA_FILE.exists():
        d = _default_data()
        DATA_FILE.write_text(json.dumps(d, indent=2))
        return d
    return json.loads(DATA_FILE.read_text())

def save_db(db: dict):
    DATA_FILE.write_text(json.dumps(db, indent=2))

def next_id(db: dict, key: str) -> int:
    db["_seq"][key] += 1
    return db["_seq"][key]

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="StreamVault", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login", auto_error=False)

# ── Auth helpers ───────────────────────────────────────────────────────────
def hash_pw(p: str) -> str:
    return hashlib.sha256(p.encode()).hexdigest()

def make_token(username: str) -> str:
    exp = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE)
    return jwt.encode({"sub": username, "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)

async def current_user(token: str = Depends(oauth2_scheme)) -> dict:
    if not token:
        raise HTTPException(401, "Not authenticated")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload["sub"]
    except Exception:
        raise HTTPException(401, "Invalid token")
    db = load_db()
    user = db["users"].get(username)
    if not user:
        raise HTTPException(401, "User not found")
    return user

async def admin_user(user=Depends(current_user)) -> dict:
    if not user.get("is_admin"):
        raise HTTPException(403, "Admin only")
    return user

# ── Models ─────────────────────────────────────────────────────────────────
class UserCreate(BaseModel):
    username: str
    password: str

class LibraryCreate(BaseModel):
    name:       str
    path:       str
    media_type: str   # video | audio | photo

class ProgressUpdate(BaseModel):
    position:  float
    completed: bool = False

class PlaylistCreate(BaseModel):
    name: str

class PlaylistAddItem(BaseModel):
    media_id: int

# ── Media scanning ─────────────────────────────────────────────────────────
def scan_library(db: dict, lib_id: str):
    """Scan a library folder and upsert media records."""
    lib = db["libraries"].get(str(lib_id))
    if not lib:
        return
    folder = Path(lib["path"])
    if not folder.exists():
        return

    media_type = lib["media_type"]
    valid_exts = (
        SUPPORTED_VIDEO if media_type == "video" else
        SUPPORTED_AUDIO if media_type == "audio" else
        SUPPORTED_IMAGE
    )

    existing = {
        m["filepath"]: mid
        for mid, m in db["media"].items()
        if m["library_id"] == lib_id
    }

    count = 0
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in valid_exts:
            continue
        fp   = str(path)
        stat = path.stat()

        if fp in existing:
            mid = existing[fp]
            db["media"][mid]["size"]     = stat.st_size
            db["media"][mid]["modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        else:
            mid = str(next_id(db, "media"))
            db["media"][mid] = {
                "id":             int(mid),
                "library_id":     lib_id,
                "media_type":     media_type,
                "filename":       path.name,
                "title":          path.stem,
                "filepath":       fp,
                "size":           stat.st_size,
                "duration":       None,
                "thumbnail_path": None,
                "modified":       datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "added_at":       datetime.utcnow().isoformat(),
            }
        count += 1

    # Remove records for files deleted from disk
    stale = [
        mid for mid, m in db["media"].items()
        if m["library_id"] == lib_id and not Path(m["filepath"]).exists()
    ]
    for mid in stale:
        del db["media"][mid]

    lib["media_count"]  = count
    lib["last_scanned"] = datetime.utcnow().isoformat()

def get_duration(filepath: str) -> Optional[float]:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", filepath],
            capture_output=True, text=True, timeout=10,
        )
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return None

def media_resp(m: dict) -> dict:
    return {
        "id":             m["id"],
        "media_id":       m["id"],
        "library_id":     m["library_id"],
        "media_type":     m["media_type"],
        "filename":       m["filename"],
        "title":          m["title"],
        "size":           m["size"],
        "duration":       m["duration"],
        "thumbnail_path": m["thumbnail_path"],
        "modified":       m["modified"],
        "added_at":       m["added_at"],
    }

# ── Serve frontend ─────────────────────────────────────────────────────────
INDEX = Path("index.html")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    if not INDEX.exists():
        return HTMLResponse("<h1>StreamVault</h1><p>index.html missing next to main.py</p>", 500)
    return HTMLResponse(INDEX.read_text())

# ── Auth ───────────────────────────────────────────────────────────────────
@app.post("/api/auth/login")
async def login(form: OAuth2PasswordRequestForm = Depends()):
    db   = load_db()
    user = db["users"].get(form.username)
    if not user or user["password_hash"] != hash_pw(form.password):
        raise HTTPException(401, "Invalid credentials")
    return {
        "access_token": make_token(user["username"]),
        "token_type":   "bearer",
        "username":     user["username"],
        "is_admin":     user.get("is_admin", False),
    }

@app.get("/api/auth/me")
async def me(user=Depends(current_user)):
    return {"username": user["username"], "is_admin": user.get("is_admin", False)}

@app.post("/api/auth/register")
async def register(body: UserCreate, _=Depends(admin_user)):
    with _db_lock:
        db = load_db()
        if body.username in db["users"]:
            raise HTTPException(400, "Username already exists")
        db["users"][body.username] = {
            "username":      body.username,
            "password_hash": hash_pw(body.password),
            "is_admin":      False,
            "created_at":    datetime.utcnow().isoformat(),
        }
        save_db(db)
    return {"username": body.username, "is_admin": False}

# ── Libraries ──────────────────────────────────────────────────────────────
@app.get("/api/libraries")
async def list_libraries(user=Depends(current_user)):
    db = load_db()
    return list(db["libraries"].values())

@app.post("/api/libraries")
async def create_library(body: LibraryCreate, _=Depends(admin_user)):
    with _db_lock:
        db     = load_db()
        lib_id = str(next_id(db, "lib"))
        lib    = {
            "id":           lib_id,
            "name":         body.name,
            "path":         body.path,
            "media_type":   body.media_type,
            "media_count":  0,
            "last_scanned": None,
            "created_at":   datetime.utcnow().isoformat(),
        }
        db["libraries"][lib_id] = lib
        scan_library(db, lib_id)
        save_db(db)
    return lib

@app.post("/api/libraries/{lib_id}/scan")
async def scan_lib(lib_id: str, _=Depends(admin_user)):
    with _db_lock:
        db = load_db()
        if lib_id not in db["libraries"]:
            raise HTTPException(404, "Library not found")
        scan_library(db, lib_id)
        save_db(db)
    return {"message": "Scan complete", "media_count": db["libraries"][lib_id]["media_count"]}

@app.delete("/api/libraries/{lib_id}")
async def delete_library(lib_id: str, _=Depends(admin_user)):
    with _db_lock:
        db = load_db()
        if lib_id not in db["libraries"]:
            raise HTTPException(404, "Library not found")
        del db["libraries"][lib_id]
        stale = [mid for mid, m in db["media"].items() if m["library_id"] == lib_id]
        for mid in stale:
            del db["media"][mid]
        save_db(db)
    return {"message": "Library deleted"}

# ── Media ──────────────────────────────────────────────────────────────────
@app.get("/api/media")
async def list_media(
    library_id: Optional[str] = None,
    media_type: Optional[str] = Query(None),
    search:     Optional[str] = None,
    limit:      int = 100,
    offset:     int = 0,
    user=Depends(current_user),
):
    db    = load_db()
    items = list(db["media"].values())

    if library_id:
        items = [m for m in items if str(m["library_id"]) == str(library_id)]
    if media_type:
        items = [m for m in items if m["media_type"] == media_type]
    if search:
        q     = search.lower()
        items = [m for m in items if q in m["title"].lower() or q in m["filename"].lower()]

    items.sort(key=lambda m: m.get("added_at", ""), reverse=True)
    total = len(items)
    page  = items[offset : offset + limit]
    return {"total": total, "items": [media_resp(m) for m in page]}

@app.get("/api/media/{media_id}")
async def get_media_item(media_id: int, user=Depends(current_user)):
    db = load_db()
    m  = db["media"].get(str(media_id))
    if not m:
        raise HTTPException(404, "Not found")
    # Lazily fetch duration via ffprobe
    if m["duration"] is None:
        dur = get_duration(m["filepath"])
        if dur:
            with _db_lock:
                db2 = load_db()
                if str(media_id) in db2["media"]:
                    db2["media"][str(media_id)]["duration"] = dur
                    save_db(db2)
            m["duration"] = dur
    return media_resp(m)

# ── Streaming ──────────────────────────────────────────────────────────────
@app.get("/api/media/{media_id}/stream")
async def stream(
    media_id: int,
    request:  Request,
    token:    Optional[str] = None,
    user=Depends(current_user),
):
    db = load_db()
    m  = db["media"].get(str(media_id))
    if not m:
        raise HTTPException(404, "Not found")
    fp = Path(m["filepath"])
    if not fp.exists():
        raise HTTPException(404, "File missing on disk")

    file_size    = fp.stat().st_size
    mime         = mimetypes.guess_type(str(fp))[0] or "application/octet-stream"
    range_header = request.headers.get("range")

    if range_header:
        start_s, end_s = range_header.replace("bytes=", "").split("-")
        start  = int(start_s)
        end    = int(end_s) if end_s else file_size - 1
        length = end - start + 1

        def _range_iter():
            with open(fp, "rb") as f:
                f.seek(start)
                rem = length
                while rem > 0:
                    chunk = f.read(min(65536, rem))
                    if not chunk:
                        break
                    rem -= len(chunk)
                    yield chunk

        return StreamingResponse(_range_iter(), status_code=206, headers={
            "Content-Range":  f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges":  "bytes",
            "Content-Length": str(length),
            "Content-Type":   mime,
        })

    def _full_iter():
        with open(fp, "rb") as f:
            while chunk := f.read(65536):
                yield chunk

    return StreamingResponse(_full_iter(), headers={
        "Content-Length": str(file_size),
        "Accept-Ranges":  "bytes",
        "Content-Type":   mime,
    })

@app.get("/api/media/{media_id}/transcode")
async def transcode(
    media_id: int,
    quality:  str = "720p",
    token:    Optional[str] = None,
    user=Depends(current_user),
):
    db = load_db()
    m  = db["media"].get(str(media_id))
    if not m:
        raise HTTPException(404, "Not found")
    fp = Path(m["filepath"])
    if not fp.exists():
        raise HTTPException(404, "File missing on disk")

    scale = {"480p": "854:480", "720p": "1280:720", "1080p": "1920:1080"}.get(quality, "1280:720")
    cmd = [
        "ffmpeg", "-i", str(fp),
        "-vf", f"scale={scale}",
        "-vcodec", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-acodec", "aac", "-b:a", "128k",
        "-f", "mp4", "-movflags", "frag_keyframe+empty_moov",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _gen():
        try:
            while chunk := proc.stdout.read(65536):
                yield chunk
        finally:
            proc.kill()

    return StreamingResponse(_gen(), media_type="video/mp4")

# ── Thumbnails ─────────────────────────────────────────────────────────────
@app.get("/api/media/{media_id}/thumbnail")
async def thumbnail(
    media_id: int,
    token:    Optional[str] = None,
    user=Depends(current_user),
):
    db = load_db()
    m  = db["media"].get(str(media_id))
    if not m:
        raise HTTPException(404, "Not found")

    # Photos: serve the file directly
    if m["media_type"] == "photo":
        fp = Path(m["filepath"])
        if not fp.exists():
            raise HTTPException(404, "File missing")
        return FileResponse(str(fp))

    # Videos: generate thumbnail with ffmpeg
    thumb = THUMB_DIR / f"{media_id}.jpg"
    if not thumb.exists():
        fp = Path(m["filepath"])
        if not fp.exists():
            raise HTTPException(404, "File missing")
        r = subprocess.run(
            ["ffmpeg", "-i", str(fp), "-ss", "00:00:05",
             "-vframes", "1", "-vf", "scale=320:-1", str(thumb)],
            capture_output=True,
        )
        if r.returncode != 0:
            raise HTTPException(500, "Thumbnail generation failed")
        with _db_lock:
            db2 = load_db()
            if str(media_id) in db2["media"]:
                db2["media"][str(media_id)]["thumbnail_path"] = str(thumb)
                save_db(db2)

    return FileResponse(str(thumb), media_type="image/jpeg")

# ── Watch progress ─────────────────────────────────────────────────────────
@app.post("/api/media/{media_id}/progress")
async def save_progress(media_id: int, body: ProgressUpdate, user=Depends(current_user)):
    key = f"{user['username']}:{media_id}"
    with _db_lock:
        db = load_db()
        db["progress"][key] = {
            "media_id":   media_id,
            "username":   user["username"],
            "position":   body.position,
            "completed":  body.completed,
            "updated_at": datetime.utcnow().isoformat(),
        }
        save_db(db)
    return {"ok": True}

@app.get("/api/history")
async def history(user=Depends(current_user)):
    db      = load_db()
    prefix  = f"{user['username']}:"
    entries = sorted(
        [v for k, v in db["progress"].items() if k.startswith(prefix)],
        key=lambda x: x["updated_at"], reverse=True,
    )
    result = []
    for p in entries:
        m = db["media"].get(str(p["media_id"]))
        if m:
            result.append({**media_resp(m), "position": p["position"], "completed": p["completed"]})
    return result

@app.get("/api/continue-watching")
async def continue_watching(user=Depends(current_user)):
    db      = load_db()
    prefix  = f"{user['username']}:"
    entries = sorted(
        [v for k, v in db["progress"].items()
         if k.startswith(prefix) and not v["completed"] and v["position"] > 5],
        key=lambda x: x["updated_at"], reverse=True,
    )
    result = []
    for p in entries[:10]:
        m = db["media"].get(str(p["media_id"]))
        if m:
            result.append({**media_resp(m), "position": p["position"], "completed": p["completed"]})
    return result

# ── Playlists ──────────────────────────────────────────────────────────────
@app.get("/api/playlists")
async def list_playlists(user=Depends(current_user)):
    db   = load_db()
    mine = sorted(
        [p for p in db["playlists"].values() if p["username"] == user["username"]],
        key=lambda p: p["created_at"],
    )
    return mine

@app.post("/api/playlists")
async def create_playlist(body: PlaylistCreate, user=Depends(current_user)):
    with _db_lock:
        db  = load_db()
        pid = str(next_id(db, "playlist"))
        pl  = {
            "id":         pid,
            "name":       body.name,
            "username":   user["username"],
            "created_at": datetime.utcnow().isoformat(),
        }
        db["playlists"][pid]      = pl
        db["playlist_items"][pid] = []
        save_db(db)
    return pl

@app.get("/api/playlists/{playlist_id}/items")
async def playlist_items(playlist_id: str, user=Depends(current_user)):
    db = load_db()
    if playlist_id not in db["playlists"]:
        raise HTTPException(404, "Playlist not found")
    ids    = db["playlist_items"].get(playlist_id, [])
    result = []
    for mid in ids:
        m = db["media"].get(str(mid))
        if m:
            result.append(media_resp(m))
    return result

@app.post("/api/playlists/{playlist_id}/items")
async def add_to_playlist(playlist_id: str, body: PlaylistAddItem, user=Depends(current_user)):
    with _db_lock:
        db = load_db()
        if playlist_id not in db["playlists"]:
            raise HTTPException(404, "Playlist not found")
        items = db["playlist_items"].get(playlist_id, [])
        if body.media_id not in items:
            items.append(body.media_id)
        db["playlist_items"][playlist_id] = items
        save_db(db)
    return {"ok": True}

@app.delete("/api/playlists/{playlist_id}")
async def delete_playlist(playlist_id: str, user=Depends(current_user)):
    with _db_lock:
        db = load_db()
        if playlist_id not in db["playlists"]:
            raise HTTPException(404, "Playlist not found")
        if db["playlists"][playlist_id]["username"] != user["username"] and not user.get("is_admin"):
            raise HTTPException(403, "Not your playlist")
        del db["playlists"][playlist_id]
        db["playlist_items"].pop(playlist_id, None)
        save_db(db)
    return {"message": "Deleted"}

# ── Admin ──────────────────────────────────────────────────────────────────
@app.get("/api/admin/users")
async def list_users(_=Depends(admin_user)):
    db = load_db()
    return [
        {"username": u["username"], "is_admin": u.get("is_admin", False), "created_at": u["created_at"]}
        for u in db["users"].values()
    ]

@app.delete("/api/admin/users/{username}")
async def delete_user(username: str, user=Depends(admin_user)):
    if username == user["username"]:
        raise HTTPException(400, "Cannot delete yourself")
    with _db_lock:
        db = load_db()
        if username not in db["users"]:
            raise HTTPException(404, "User not found")
        del db["users"][username]
        save_db(db)
    return {"message": "User deleted"}

# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
