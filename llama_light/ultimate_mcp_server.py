#!/usr/bin/env python3
"""
Professional MCP Server v4.0 - Ultimate AI Operations Suite
Complete browser control, remote access, AI integration, and more
"""

import asyncio
import json
import logging
import os
import sys
import re
import time
import subprocess
import hashlib
import urllib.request
import urllib.parse
import urllib.error
import socket
import base64
import mimetypes
import zipfile
import tarfile
import io
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Generator
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from html.parser import HTMLParser
from collections import defaultdict

import uvicorn
from fastapi import FastAPI, Request, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Optional dependencies
try:
    from ddgs import DDGS as DDGS_LIB
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
try:
    from ddgs import DDGS as DDGS_LIB
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
try:
    from ddgs import DDGS as DDGS_LIB
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
try:
    from ddgs import DDGS as DDGS_LIB
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    SMTP_AVAILABLE = True
except ImportError:
    SMTP_AVAILABLE = False

try:
    import ssh2
    SSH_AVAILABLE = True
except ImportError:
    SSH_AVAILABLE = False

try:
    import psycopg2
    import pymongo
    import redis
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import openai
    import chromadb
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False

# --- Configuration ---
SERVER_VERSION = "4.0.0"
SERVER_NAME = "Ultimate MCP Server"
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '')
GOOGLE_CX = os.environ.get('GOOGLE_CX', '')
UPLOAD_DIR = Path.home() / ".mcp_server" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR = Path.home() / ".mcp_server" / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ultimate-mcp")

app = FastAPI(title=SERVER_NAME, description="Ultimate AI Operations Suite", version=SERVER_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Browser/HTML Parser (for scraping)
# ============================================================================

class WebPageParser(HTMLParser):
    """Parse web pages and extract structured data."""
    def __init__(self):
        super().__init__()
        self.links = []
        self.images = []
        self.headings = []
        self.paragraphs = []
        self.tables = []
        self.meta = {}
        self.current_tag = None
        self.current_attrs = {}
        self.in_table = False
        self.table_data = []
        self.current_row = []
        self.in_cell = False
        self.cell_data = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        self.current_tag = tag
        self.current_attrs = attrs_dict
        
        if tag == 'a' and attrs_dict.get('href'):
            self.links.append({"url": attrs_dict['href'], "text": "", "attrs": attrs_dict})
        elif tag == 'img' and attrs_dict.get('src'):
            self.images.append({"src": attrs_dict['src'], "alt": attrs_dict.get('alt', '')})
        elif tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            self.headings.append({"level": int(tag[1]), "text": ""})
        elif tag == 'p':
            self.paragraphs.append({"text": ""})
        elif tag == 'table':
            self.in_table = True
            self.table_data = []
        elif tag == 'tr':
            self.current_row = []
        elif tag in ('td', 'th') and self.in_table:
            self.in_cell = True
            self.cell_data = []
        elif tag in ('meta'):
            name = attrs_dict.get('name', attrs_dict.get('property', ''))
            content = attrs_dict.get('content', '')
            if name and content:
                self.meta[name] = content

    def handle_endtag(self, tag):
        if tag == 'a':
            if self.links:
                self.links[-1]['text'] = self.get_last_text()
        elif tag == 'img':
            pass
        elif tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            idx = len(self.headings) - 1
            if idx >= 0:
                self.headings[idx]['text'] = self.get_last_text()
        elif tag == 'p':
            if self.paragraphs:
                self.paragraphs[-1]['text'] = self.get_last_text()
        elif tag == 'td' or tag == 'th':
            if self.in_cell:
                self.in_cell = False
                self.current_row.append(''.join(self.cell_data))
                self.cell_data = []
        elif tag == 'tr' and self.in_table:
            if self.current_row:
                self.table_data.append(self.current_row)
            self.current_row = []
        elif tag == 'table' and self.in_table:
            self.in_table = False
            if self.table_data:
                self.tables.append(self.table_data)
            self.table_data = []

    def handle_data(self, data):
        if self.current_tag == 'a' and self.links:
            self.links[-1]['text'] += data
        elif self.current_tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            idx = len(self.headings) - 1
            if idx >= 0:
                self.headings[idx]['text'] += data
        elif self.current_tag == 'p' and self.paragraphs:
            self.paragraphs[-1]['text'] += data
        elif self.in_cell:
            self.cell_data.append(data)

    def get_last_text(self):
        return ''.join(self.cell_data) if self.cell_data else ''


def fetch_url(url: str, headers: Optional[Dict] = None, timeout: int = 15, follow_redirects: int = 5) -> Dict[str, Any]:
    """Fetch a URL and return HTML/content."""
    if not headers:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
    
    try:
        req = urllib.request.Request(url, headers=headers, method='GET')
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            encoding = resp.headers.get_content_charset() or 'utf-8'
            html = resp.read().decode(encoding, errors='replace')
            content_type = resp.headers.get('Content-Type', '')
            
            # Parse HTML
            parsed = {}
            if 'text/html' in content_type:
                parser = WebPageParser()
                parser.feed(html)
                parsed = {
                    "links": [{"url": l["url"], "text": l["text"]} for l in parser.links if l["text"]][:100],
                    "images": parser.images[:50],
                    "headings": [h for h in parser.headings if h["text"]],
                    "paragraphs": [{"text": p["text"]} for p in parser.paragraphs if p["text"]][:50],
                    "tables": parser.tables[:10],
                    "meta": parser.meta
                }
            
            return {
                "success": True,
                "url": url,
                "status_code": resp.status if hasattr(resp, 'status') else 200,
                "content_type": content_type,
                "content": html[:100000],  # Limit content size
                "parsed": parsed,
                "headers": {k: v for k, v in resp.headers.items()},
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        return {"success": False, "error": str(e), "url": url}


def scrape_page(url: str, selectors: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Scrape specific elements from a page."""
    if not BS4_AVAILABLE:
        return {"success": False, "error": "BeautifulSoup not installed. pip install beautifulsoup4"}
    
    try:
        page = fetch_url(url)
        if not page["success"]:
            return page
        
        html = page["content"]
        soup = BeautifulSoup(html, 'html.parser')
        
        results = {}
        if not selectors:
            # Extract all text
            results = {"text": soup.get_text(separator='\n', strip=True)[:10000]}
        else:
            for name, selector in selectors.items():
                elements = soup.select(selector)
                if elements:
                    results[name] = [
                        e.get_text(strip=True) if e.get_text(strip=True) else 
                        e.get('href') if e.name == 'a' else
                        e.get('src') if e.name == 'img' else
                        e.text
                        for e in elements
                    ][:20]
        
        return {"success": True, "url": url, "data": results, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}


def extract_table(url: str) -> Dict[str, Any]:
    """Extract tables from a webpage."""
    try:
        page = fetch_url(url)
        if not page["success"]:
            return page
        
        soup = BeautifulSoup(page["content"], 'html.parser')
        tables = soup.find_all('table')
        extracted = []
        
        for i, table in enumerate(tables[:10]):
            rows = table.find_all(['tr'])
            data = []
            for row in rows:
                cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
                if cells:
                    data.append(cells)
            if data:
                extracted.append({"table_index": i, "headers": data[0], "rows": data[1:]})
        
        return {"success": True, "url": url, "tables": extracted, "count": len(extracted)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def find_links(url: str, pattern: str = None, limit: int = 50) -> Dict[str, Any]:
    """Find all links on a page."""
    try:
        page = fetch_url(url)
        if not page["success"]:
            return page
        
        soup = BeautifulSoup(page["content"], 'html.parser')
        links = soup.find_all('a', href=True)
        
        results = []
        for link in links[:limit]:
            href = link['href']
            href = urllib.parse.urljoin(url, href) if not href.startswith('http') else href
            text = link.get_text(strip=True)
            
            if pattern and not re.search(pattern, href):
                continue
            
            results.append({
                "url": href,
                "text": text[:200],
                "title": link.get('title', '')
            })
        
        return {"success": True, "url": url, "links": results, "count": len(results)}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# Search Engines
# ============================================================================

def scrape_google_duckduckgo(query: str, num: int = 5) -> Optional[List[Dict]]:
    """Scrape Google via DuckDuckGo HTML proxy."""
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode()
        
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        for result in soup.select('div.result')[:num]:
            title_elem = result.select_one('a.result__title')
            link_elem = result.select_one('a.result__snippet')
            snippet = result.select_one('a.result__snippet')
            
            title = title_elem.get_text(strip=True) if title_elem else ''
            link = title_elem.get('href', '') if title_elem else ''
            snippet_text = snippet.get_text(strip=True) if snippet else ''
            
            if title:
                results.append({"title": title, "url": link, "snippet": snippet_text})
        
        return results if results else None
    except Exception as e:
        logger.error(f"Scrape failed: {e}")
        return None


def google_custom_search(query: str, num_results: int = 5) -> Optional[List[Dict[str, str]]]:
    """Google Custom Search API."""
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "":
        return None
        return None
    url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={GOOGLE_CX}&q={urllib.parse.quote(query)}&num={min(num_results, 10)}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'UltimateMCP/4.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            return [{"title": i.get('title', ''), "url": i.get('link', ''),
                     "snippet": i.get('snippet', '')} for i in data.get('items', [])]
    except Exception as e:
        logger.error(f"Google API failed: {e}")
        return None


def duckduckgo_search(query: str, max_results: int = 5) -> Optional[List[Dict]]:
    """DuckDuckGo search."""
    if not DDGS_AVAILABLE:
        return None
    try:
            ddgs = DDGS_LIB()
            results = ddgs.text(query, max_results=max_results)
            return [{"title": r.get('title', ''  ), "url": r.get('href', ''  ),
                 "snippet": r.get('body', '')} for r in results]
    except Exception as e:
        logger.error(f"DDG search failed: {e}")
        return None


def perform_search(query: str, engine: str = "auto", max_results: int = 5) -> Dict[str, Any]:
    """Unified search across all engines."""
    if not query:
        return {"success": False, "error": "Query is required"}
    
    # Auto-select engine
    if engine == "auto":
        if GOOGLE_API_KEY and GOOGLE_CX:
            engine = "google"
        else:
            engine = "scrape_google"
    
    # Try Google API first (best quality)
    if engine == "google":
        results = google_custom_search(query, max_results)
        if results:
            return {"success": True, "engine": "google", "query": query,
                    "results": results, "total_results": len(results)}
        return {"success": False, "error": "Google API not configured"}
    
    # Try scrape Google
    if engine == "scrape_google":
        results = scrape_google_duckduckgo(query, max_results)
        if results:
            return {"success": True, "engine": "scrape_google", "query": query,
                    "results": results, "total_results": len(results)}
    
    # Fall back to DuckDuckGo
    if engine in ("duckduckgo", "auto"):
        results = duckduckgo_search(query, max_results)
        if results:
            return {"success": True, "engine": "duckduckgo", "query": query,
                    "results": results, "total_results": len(results)}
    
    return {"success": False, "error": "All search engines failed"}


# ============================================================================
# System & Terminal Tools
# ============================================================================

def safe_execute(command: str, timeout: int = 30, shell: bool = True,
                 capture_output: bool = True, text: bool = True,
                 env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Execute shell command safely."""
    try:
        result = subprocess.run(command, shell=shell, capture_output=capture_output,
                                text=text, timeout=timeout, env=env, input=None)
        return {"stdout": result.stdout if capture_output else "",
                "stderr": result.stderr if capture_output else "",
                "return_code": result.returncode, "success": result.returncode == 0,
                "timestamp": datetime.now().isoformat()}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Timed out after {timeout}s",
                "return_code": -1, "success": False, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "return_code": -1, "success": False,
                "timestamp": datetime.now().isoformat()}


def read_file_safe(path: str, encoding: str = "utf-8", errors: str = "ignore") -> Dict[str, Any]:
    try:
        file_path = Path(path)
        if not file_path.exists():
            return {"success": False, "error": f"File not found: {path}"}
        size = file_path.stat().st_size
        if size > 50 * 1024 * 1024:
            return {"success": False, "error": "File too large (>50MB)"}
        with open(file_path, 'r', encoding=encoding, errors=errors) as f:
            content = f.read()
        return {"success": True, "path": str(file_path.absolute()), "content": content,
                "size_bytes": size, "lines": content.count('\n') + 1, "encoding": encoding,
                "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}


def write_file_safe(path: str, content: str, create_dirs: bool = True) -> Dict[str, Any]:
    try:
        file_path = Path(path)
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {"success": True, "path": str(file_path.absolute()),
                "bytes_written": len(content), "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}


def search_codebase(directory: str, pattern: str,
                    extensions: Optional[List[str]] = None, max_results: int = 100) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            f"grep -r -l '{pattern}' '{directory}' 2>/dev/null | head -{max_results}",
            shell=True, capture_output=True, text=True, timeout=15)
        files = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        return {"success": True, "files_found": files, "total_files": len(files),
                "pattern": pattern, "directory": directory, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}


def git_operation(repo_path: str, operation: str, **kwargs) -> Dict[str, Any]:
    try:
        commands = {
            "status": f"cd '{repo_path}' && git status --short",
            "log": f"cd '{repo_path}' && git log --oneline -10",
            "diff": f"cd '{repo_path}' && git diff",
            "branches": f"cd '{repo_path}' && git branch -a",
            "remote": f"cd '{repo_path}' && git remote -v",
        }
        if operation == "commit":
            cmd = f"cd '{repo_path}' && git add . && git commit -m '{kwargs.get('message', 'Auto-commit')}'"
        elif operation == "push":
            cmd = f"cd '{repo_path}' && git push"
        else:
            cmd = commands.get(operation, f"cd '{repo_path}' && git {operation}")
        return safe_execute(cmd, timeout=30)
    except Exception as e:
        return {"success": False, "error": str(e)}


def scan_secrets(content: str) -> Dict[str, Any]:
    patterns = {
        "API Keys": r'(?:api[_-]?key|apikey|api[_-]?secret)[\s]*[:=][\s]*[\'"]?([a-zA-Z0-9]{20,})[\'"]?',
        "Passwords": r'(?:password|passwd|pwd)[\s]*[:=][\s]*[\'"]?([^\'"\\s]{8,})[\'"]?',
        "AWS Keys": r'(?:AKIA[0-9A-Z]{16})',
        "Private Keys": r'-----BEGIN\s+(?:RSA|DSA|EC)\s+PRIVATE\s+KEY-----',
        "Tokens": r'(?:bearer|token)[\s]*[:=][\s]*[\'"]?([a-zA-Z0-9\-_.]{20,})[\'"]?',
    }
    findings = []
    for name, pattern in patterns.items():
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            findings.append({"type": name, "count": len(matches),
                             "samples": [m[0] if isinstance(m, tuple) else m for m in matches[:3]]})
    return {"success": True, "findings": findings,
            "total_findings": sum(f["count"] for f in findings),
            "timestamp": datetime.now().isoformat()}


def analyze_code_quality(file_path: str) -> Dict[str, Any]:
    try:
        fc = read_file_safe(file_path)
        if not fc["success"]:
            return fc
        lines = fc["content"].split('\n')
        issues_list = []
        for i, l in enumerate(lines):
            if len(l) > 120:
                issues_list.append({"line": i+1, "issue": "Long line", "detail": str(len(l))})
            elif '\t' in l:
                issues_list.append({"line": i+1, "issue": "Tab", "detail": "Tab character"})
        metrics = {
            "total_lines": len(lines),
            "code_lines": sum(1 for l in lines if l.strip() and not l.strip().startswith('#')),
            "comment_lines": sum(1 for l in lines if l.strip().startswith('#')),
            "blank_lines": sum(1 for l in lines if not l.strip()),
            "max_line_length": max((len(l) for l in lines), default=0)
        }
        return {
            "success": True, "path": file_path, "metrics": metrics,
            "issues": issues_list[:20],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# File Management
# ============================================================================

def list_directory(path: str = ".", depth: int = 1) -> Dict[str, Any]:
    """List directory with depth control."""
    try:
        dir_path = Path(path)
        if not dir_path.exists():
            return {"success": False, "error": f"Directory not found: {path}"}
        
        items = []
        for p in sorted(dir_path.iterdir()):
            try:
                stat = p.stat()
                items.append({
                    "name": p.name,
                    "path": str(p),
                    "is_dir": p.is_dir(),
                    "is_file": p.is_file(),
                    "size": stat.st_size if p.is_file() else None,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "permissions": oct(stat.st_mode)[-3:]
                })
            except:
                pass
        
        return {"success": True, "path": str(dir_path.absolute()),
                "items": items, "total_items": len(items)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def copy_file(source: str, destination: str) -> Dict[str, Any]:
    try:
        src = Path(source)
        dst = Path(destination)
        if not src.exists():
            return {"success": False, "error": f"Source not found: {source}"}
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        return {"success": True, "source": str(src.absolute()),
                "destination": str(dst.absolute()),
                "size": src.stat().st_size}
    except Exception as e:
        return {"success": False, "error": str(e)}


def move_file(source: str, destination: str) -> Dict[str, Any]:
    try:
        src = Path(source)
        dst = Path(destination)
        if not src.exists():
            return {"success": False, "error": f"Source not found: {source}"}
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        return {"success": True, "source": str(src.absolute()),
                "destination": str(dst.absolute())}
    except Exception as e:
        return {"success": False, "error": str(e)}


def delete_file(path: str) -> Dict[str, Any]:
    try:
        file_path = Path(path)
        if not file_path.exists():
            return {"success": False, "error": f"Not found: {path}"}
        if file_path.is_dir():
            subprocess.run(['rm', '-rf', str(file_path)], capture_output=True)
        else:
            file_path.unlink()
        return {"success": True, "deleted": str(file_path)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def create_directory(path: str) -> Dict[str, Any]:
    try:
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return {"success": True, "path": str(dir_path.absolute())}
    except Exception as e:
        return {"success": False, "error": str(e)}


def compress_files(files: List[str], output: str) -> Dict[str, Any]:
    """Create zip archive."""
    try:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                fp = Path(f)
                if fp.exists():
                    zf.write(fp, fp.name)
        return {"success": True, "archive": str(output_path),
                "file_count": len(files), "size": output_path.stat().st_size}
    except Exception as e:
        return {"success": False, "error": str(e)}


def upload_file(file_data: str, filename: str, mime_type: str = None) -> Dict[str, Any]:
    """Upload a file (base64 encoded)."""
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = UPLOAD_DIR / filename
        decoded = base64.b64decode(file_data)
        file_path.write_bytes(decoded)
        return {"success": True, "path": str(file_path), "size": len(decoded),
                "filename": filename, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}


def download_file(file_path: str, range_start: int = None, range_end: int = None) -> Dict[str, Any]:
    """Download a file as base64."""
    try:
        fp = Path(file_path)
        if not fp.exists():
            return {"success": False, "error": f"Not found: {file_path}"}
        
        data = fp.read_bytes()
        mime_type = mimetypes.guess_type(str(fp))[0] or 'application/octet-stream'
        
        return {
            "success": True,
            "filename": fp.name,
            "size": len(data),
            "mime_type": mime_type,
            "content_base64": base64.b64encode(data).decode(),
            "download_url": f"/files/download/{urllib.parse.quote(str(fp))}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# SSH / Remote Control
# ============================================================================

def ssh_execute(host: str, port: int = 22, username: str = None, 
                password: str = None, key_file: str = None, 
                command: str = "uname -a") -> Dict[str, Any]:
    """Execute command via SSH."""
    try:
        cmd = f"ssh {'-o StrictHostKeyChecking=no ' if not password else ''}"
        if password:
            cmd += f"-o PreferredAuthentications=password -o PubkeyAuthentication=no "
        if key_file:
            cmd += f"-i {key_file} "
        cmd += f"-p {port} {username}@{host} '{command}'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
            "host": host,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def ssh_execute_batch(host: str, commands: List[str], username: str = None,
                      password: str = None) -> Dict[str, Any]:
    """Execute multiple commands via SSH."""
    try:
        results = []
        for cmd in commands:
            ssh_cmd = f"ssh -o StrictHostKeyChecking=no -p 22 {username}@{host} '{cmd}'"
            if password:
                ssh_cmd = f"sshpass -p '{password}' {ssh_cmd}"
            r = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=30)
            results.append({"command": cmd, "stdout": r.stdout, "stderr": r.stderr,
                           "return_code": r.returncode, "success": r.returncode == 0})
        return {"success": True, "host": host, "results": results}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# API Testing
# ============================================================================

def test_api(url: str, method: str = "GET", headers: Dict = None, 
             body: str = None, timeout: int = 10) -> Dict[str, Any]:
    """Test any REST API endpoint."""
    try:
        import requests as req
        hdrs = headers or {}
        hdrs['User-Agent'] = 'UltimateMCP/4.0'
        
        if method.upper() == 'GET':
            resp = req.get(url, headers=hdrs, timeout=timeout)
        elif method.upper() == 'POST':
            resp = req.post(url, headers=hdrs, data=body if body else {}, timeout=timeout)
        elif method.upper() == 'PUT':
            resp = req.put(url, headers=hdrs, data=body if body else {}, timeout=timeout)
        elif method.upper() == 'DELETE':
            resp = req.delete(url, headers=hdrs, timeout=timeout)
        elif method.upper() == 'PATCH':
            resp = req.patch(url, headers=hdrs, data=body if body else {}, timeout=timeout)
        else:
            return {"success": False, "error": f"Unsupported method: {method}"}
        
        try:
            body_json = resp.json()
        except:
            body_json = resp.text[:5000] if resp.text else None
        
        return {
            "success": True,
            "url": url,
            "method": method.upper(),
            "status_code": resp.status_code,
            "headers": {k: v for k, v in resp.headers.items()},
            "body": body_json,
            "time_ms": resp.elapsed.total_seconds() * 1000,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# Email
# ============================================================================

def send_email(smtp_server: str, smtp_port: int = 587, username: str = None, password: str = None,
               to: str = None, subject: str = None, body: str = None, html: bool = False,
               attachments: List[str] = None) -> Dict[str, Any]:
    if not SMTP_AVAILABLE:
        return {"success": False, "error": "SMTP not available"}
        return {"success": False, "error": "SMTP not available"}
    try:
        msg = MIMEMultipart()
        msg['From'] = username
        msg['To'] = to
        msg['Subject'] = subject
        
        if html:
            msg.attach(MIMEText(body, 'html'))
        else:
            msg.attach(MIMEText(body, 'plain'))
        
        if attachments:
            for filepath in attachments:
                with open(filepath, 'rb') as f:
                    msg.attach(f)
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(username, password)
        server.sendmail(username, to, msg.as_string())
        server.quit()
        
        return {"success": True, "message": "Email sent successfully", "to": to}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# Monitoring & Health
# ============================================================================

def system_info() -> Dict[str, Any]:
    """Get comprehensive system info."""
    try:
        import platform
        import psutil
        
        info = {
            "system": platform.system(),
            "node": platform.node(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "pid": os.getpid(),
            "cpu_count": psutil.cpu_count(),
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
        }
        
        mem = psutil.virtual_memory()
        info["memory"] = {
            "total_gb": round(mem.total / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
            "percent_used": mem.percent
        }
        
        disk = psutil.disk_usage('/')
        info["disk"] = {
            "total_gb": round(disk.total / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "percent_used": disk.percent
        }
        
        net = psutil.net_io_counters()
        info["network"] = {
            "bytes_sent": net.bytes_sent,
            "bytes_recv": net.bytes_recv,
            "packets_sent": net.packets_sent,
            "packets_recv": net.packets_recv
        }
        
        proc = psutil.process_mapping(os.getpid()) if hasattr(psutil, 'process_mapping') else {}
        info["process"] = {"pid": os.getpid(), "name": "ultimate-mcp", "status": "running"}
        
        return {"success": True, "info": info}
    except Exception as e:
        return {"success": False, "error": str(e)}


def process_list(filter_pattern: str = None) -> Dict[str, Any]:
    """List running processes."""
    try:
        import psutil
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'username', 'status', 'cpu_percent', 'memory_percent']):
            try:
                p = proc.info
                if filter_pattern and filter_pattern.lower() not in p['name'].lower():
                    continue
                processes.append(p)
            except:
                pass
        return {"success": True, "count": len(processes), "processes": processes[:50]}
    except Exception as e:
        return {"success": False, "error": str(e)}


def process_kill(pid: int, signal: int = 9) -> Dict[str, Any]:
    """Kill a process."""
    try:
        import psutil
        proc = psutil.Process(pid)
        proc.send_signal(signal)
        return {"success": True, "killed": pid, "signal": signal}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# Database Operations
# ============================================================================

def db_query(db_type: str = "sqlite", connection_string: str = None,
             query: str = None, limit: int = 100) -> Dict[str, Any]:
    """Query a database."""
    if not DB_AVAILABLE:
        return {"success": False, "error": "Database drivers not installed"}
    
    try:
        if db_type == "sqlite":
            if not connection_string or not connection_string.endswith('.db'):
                connection_string = ":memory:" if not connection_string else connection_string
            conn = sqlite3.connect(connection_string)
            cursor = conn.cursor()
        else:
            return {"success": False, "error": f"DB type '{db_type}' not yet implemented"}
        
        cursor.execute(query)
        
        if query.strip().upper().startswith('SELECT'):
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchmany(limit)
            return {"success": True, "columns": columns,
                    "data": [dict(zip(columns, row)) for row in rows],
                    "row_count": len(rows)}
        else:
            conn.commit()
            return {"success": True, "message": f"Rows affected: {cursor.rowcount}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        try: conn.close()
        except: pass


# ============================================================================
# Cron / Task Scheduler
# ============================================================================

tasks = defaultdict(lambda: {"enabled": False, "command": "", "schedule": "", "created": ""})

def add_task(name: str, command: str, schedule_type: str = "cron", 
             schedule_expr: str = "* * * * *") -> Dict[str, Any]:
    """Add a scheduled task."""
    tasks[name] = {
        "enabled": True,
        "command": command,
        "schedule_type": schedule_type,
        "schedule_expr": schedule_expr,
        "created": datetime.now().isoformat()
    }
    return {"success": True, "message": f"Task '{name}' added", "task": tasks[name]}


def list_tasks() -> Dict[str, Any]:
    """List all scheduled tasks."""
    return {"success": True, "tasks": dict(tasks), "count": len(tasks)}


def remove_task(name: str) -> Dict[str, Any]:
    """Remove a scheduled task."""
    if name in tasks:
        del tasks[name]
        return {"success": True, "message": f"Task '{name}' removed"}
    return {"success": False, "error": f"Task '{name}' not found"}


# ============================================================================
# MCP Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": SERVER_VERSION, "timestamp": datetime.now().isoformat()}


@app.get("/tools")
async def list_tools():
    return {
        "tools": [
            {"name": "execute_command", "category": "system", "description": "Execute shell command",
             "params": {"command": "str", "timeout": "int", "shell": "bool"}},
            {"name": "read_file", "category": "files", "description": "Read file safely",
             "params": {"path": "str", "encoding": "str"}},
            {"name": "write_file", "category": "files", "description": "Write to file",
             "params": {"path": "str", "content": "str", "create_dirs": "bool"}},
            {"name": "list_directory", "category": "files", "description": "List directory contents",
             "params": {"path": "str", "depth": "int"}},
            {"name": "copy_file", "category": "files", "description": "Copy file",
             "params": {"source": "str", "destination": "str"}},
            {"name": "move_file", "category": "files", "description": "Move/rename file",
             "params": {"source": "str", "destination": "str"}},
            {"name": "delete_file", "category": "files", "description": "Delete file",
             "params": {"path": "str"}},
            {"name": "create_directory", "category": "files", "description": "Create directory",
             "params": {"path": "str"}},
            {"name": "compress_files", "category": "files", "description": "Create zip archive",
             "params": {"files": "list", "output": "str"}},
            {"name": "upload_file", "category": "files", "description": "Upload file (base64)",
             "params": {"file_data": "str", "filename": "str", "mime_type": "str"}},
            {"name": "download_file", "category": "files", "description": "Download file as base64",
             "params": {"file_path": "str"}},
            {"name": "search_codebase", "category": "dev", "description": "Search codebase for patterns",
             "params": {"directory": "str", "pattern": "str", "extensions": "list", "max_results": "int"}},
            {"name": "git_operation", "category": "dev", "description": "Git operations (status, log, diff, commit, push)",
             "params": {"repo_path": "str", "operation": "str", "message": "str"}},
            {"name": "scan_secrets", "category": "security", "description": "Scan content for secrets/credentials",
             "params": {"content": "str"}},
            {"name": "analyze_code", "category": "dev", "description": "Code quality analysis",
             "params": {"file_path": "str"}},
            {"name": "web_fetch", "category": "web", "description": "Fetch URL and parse HTML",
             "params": {"url": "str", "timeout": "int"}},
            {"name": "scrape_page", "category": "web", "description": "Scrape specific elements with CSS selectors",
             "params": {"url": "str", "selectors": "dict"}},
            {"name": "extract_table", "category": "web", "description": "Extract all tables from page",
             "params": {"url": "str"}},
            {"name": "find_links", "category": "web", "description": "Find all links on page",
             "params": {"url": "str", "pattern": "str", "limit": "int"}},
            {"name": "web_search", "category": "search", "description": "Unified search (auto/google/duckduckgo/scrape_google)",
             "params": {"query": "str", "engine": "str", "max_results": "int"}},
            {"name": "google_search", "category": "search", "description": "Google Custom Search API",
             "params": {"query": "str", "max_results": "int"}},
            {"name": "ssh_execute", "category": "remote", "description": "Execute command via SSH",
             "params": {"host": "str", "command": "str", "username": "str", "password": "str"}},
            {"name": "ssh_execute_batch", "category": "remote", "description": "Execute multiple commands via SSH",
             "params": {"host": "str", "commands": "list", "username": "str"}},
            {"name": "test_api", "category": "api", "description": "Test REST API endpoint",
             "params": {"url": "str", "method": "str", "headers": "dict", "body": "str"}},
            {"name": "send_email", "category": "email", "description": "Send email via SMTP",
             "params": {"smtp_server": "str", "username": "str", "password": "str",
                       "to": "str", "subject": "str", "body": "str"}},
            {"name": "db_query", "category": "database", "description": "Execute SQL query",
             "params": {"db_type": "str", "connection_string": "str", "query": "str", "limit": "int"}},
            {"name": "add_task", "category": "scheduler", "description": "Add a scheduled task",
             "params": {"name": "str", "command": "str", "schedule_type": "str", "schedule_expr": "str"}},
            {"name": "list_tasks", "category": "scheduler", "description": "List all tasks",
             "params": {}},
            {"name": "remove_task", "category": "scheduler", "description": "Remove a scheduled task",
             "params": {"name": "str"}},
            {"name": "system_info", "category": "monitoring", "description": "Comprehensive system info",
             "params": {}},
            {"name": "process_list", "category": "monitoring", "description": "List running processes",
             "params": {"filter": "str"}},
            {"name": "process_kill", "category": "monitoring", "description": "Kill a process",
             "params": {"pid": "int", "signal": "int"}},
            {"name": "browser_start", "category": "browser", "description": "Start browser session",
             "params": {}},
            {"name": "browser_navigate", "category": "browser", "description": "Navigate to URL",
             "params": {"session_id": "str", "url": "str"}},
            {"name": "browser_screenshot", "category": "browser", "description": "Take screenshot",
             "params": {"session_id": "str", "full_page": "bool"}},
            {"name": "browser_get_text", "category": "browser", "description": "Get page text",
             "params": {"session_id": "str", "selector": "str"}},
            {"name": "browser_click", "category": "browser", "description": "Click element",
             "params": {"session_id": "str", "selector": "str"}},
            {"name": "browser_fill", "category": "browser", "description": "Fill form field",
             "params": {"session_id": "str", "selector": "str", "value": "str"}},
            {"name": "browser_wait_for_selector", "category": "browser", "description": "Wait for element",
             "params": {"session_id": "str", "selector": "str"}},
            {"name": "browser_list_sessions", "category": "browser", "description": "List browser sessions",
             "params": {}},
            {"name": "browser_close_session", "category": "browser", "description": "Close browser session",
             "params": {"session_id": "str"}},
            {"name": "terminal_exec", "category": "terminal", "description": "Execute command in terminal",
             "params": {"command": "str"}},
            {"name": "terminal_list", "category": "terminal", "description": "List terminal sessions",
             "params": {}},
            {"name": "terminal_kill", "category": "terminal", "description": "Kill terminal session",
             "params": {"session_id": "str"}},
            {"name": "websocket_terminal", "category": "terminal", "description": "Interactive terminal (WebSocket)",
             "params": {}},
        ],
        "total_tools": 35
    }


# ===== System Tools =====
@app.post("/tools/execute_command")
async def execute_command_endpoint(request: Request):
    try:
        body = await request.json()
        return safe_execute(body["command"], timeout=body.get("timeout", 30), shell=body.get("shell", True))
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/tools/read_file")
async def read_file_endpoint(request: Request):
    try:
        body = await request.json()
        return read_file_safe(body["path"], body.get("encoding", "utf-8"))
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/tools/write_file")
async def write_file_endpoint(request: Request):
    try:
        body = await request.json()
        return write_file_safe(body["path"], body.get("content", ""), body.get("create_dirs", True))
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/tools/list_directory")
async def list_directory_endpoint(request: Request):
    try:
        body = await request.json()
        return list_directory(body.get("path", "."), body.get("depth", 1))
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/tools/copy_file")
async def copy_file_endpoint(request: Request):
    try:
        body = await request.json()
        return copy_file(body["source"], body["destination"])
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/tools/move_file")
async def move_file_endpoint(request: Request):
    try:
        body = await request.json()
        return move_file(body["source"], body["destination"])
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/tools/delete_file")
async def delete_file_endpoint(request: Request):
    try:
        body = await request.json()
        return delete_file(body["path"])
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/tools/create_directory")
async def create_directory_endpoint(request: Request):
    try:
        body = await request.json()
        return create_directory(body["path"])
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/tools/compress_files")
async def compress_files_endpoint(request: Request):
    try:
        body = await request.json()
        return compress_files(body["files"], body["output"])
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/tools/upload_file")
async def upload_file_endpoint(request: Request):
    try:
        body = await request.json()
        return upload_file(body["file_data"], body["filename"], body.get("mime_type"))
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/tools/download_file")
async def download_file_endpoint(request: Request):
    try:
        body = await request.json()
        return download_file(body["file_path"])
    except Exception as e:
        return {"success": False, "error": str(e)}


# ===== Dev Tools =====
@app.post("/tools/search_codebase")
async def search_codebase_endpoint(request: Request):
    try:
        body = await request.json()
        return search_codebase(body["directory"], body["pattern"],
                               body.get("extensions"), body.get("max_results", 100))
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/tools/git_operation")
async def git_operation_endpoint(request: Request):
    try:
        body = await request.json()
        return git_operation(body["repo_path"], body["operation"], **body)
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/tools/scan_secrets")
async def scan_secrets_endpoint(request: Request):
    try:
        body = await request.json()
        return scan_secrets(body["content"])
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/tools/analyze_code")
async def analyze_code_endpoint(request: Request):
    try:
        body = await request.json()
        return analyze_code_quality(body["file_path"])
    except Exception as e:
        return {"success": False, "error": str(e)}


# ===== Web Tools =====
@app.post("/tools/web_fetch")
async def web_fetch_endpoint(request: Request):
    try:
        body = await request.json()
        return fetch_url(body["url"], timeout=body.get("timeout", 15))
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/tools/scrape_page")
async def scrape_page_endpoint(request: Request):
    try:
        body = await request.json()
        return scrape_page(body["url"], body.get("selectors"))
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/tools/extract_table")
async def extract_table_endpoint(request: Request):
    try:
        body = await request.json()
        return extract_table(body["url"])
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/tools/find_links")
async def find_links_endpoint(request: Request):
    try:
        body = await request.json()
        return find_links(body["url"], body.get("pattern"), body.get("limit", 50))
    except Exception as e:
        return {"success": False, "error": str(e)}


# ===== Search Tools =====
@app.post("/tools/web_search")
async def web_search_endpoint(request: Request):
    try:
        body = await request.json()
        return perform_search(body["query"], body.get("engine", "auto"), body.get("max_results", 5))
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/tools/google_search")
async def google_search_endpoint(request: Request):
    try:
        body = await request.json()
        results = google_custom_search(body["query"], body.get("max_results", 5))
        if results is None:
            return {"success": False, "error": "Google API not configured. Set GOOGLE_API_KEY and GOOGLE_CX env vars."}
        return {"success": True, "query": body["query"], "engine": "google",
                "results": results, "total_results": len(results)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/tools/ssh_execute")
async def ssh_execute_endpoint(request: Request):
    try:
        body = await request.json()
        return ssh_execute(body["host"], body.get("port", 22), body.get("username"),
                           body.get("password"), body.get("key_file"), body["command"])
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/tools/ssh_execute_batch")
async def ssh_execute_batch_endpoint(request: Request):
    try:
        body = await request.json()
        return ssh_execute_batch(body["host"], body["commands"], body.get("username"), body.get("password"))
    except Exception as e:
        return {"success": False, "error": str(e)}


# ===== API Tools =====
@app.post("/tools/test_api")
async def test_api_endpoint(request: Request):
    try:
        body = await request.json()
        return test_api(body["url"], body.get("method", "GET"), body.get("headers"),
                       body.get("body"), body.get("timeout", 10))
    except Exception as e:
        return {"success": False, "error": str(e)}


# ===== Email Tools =====
@app.post("/tools/send_email")
async def send_email_endpoint(request: Request):
    try:
        body = await request.json()
        return send_email(body["smtp_server"], body.get("smtp_port", 587),
                          body["username"], body["password"],
                          body["to"], body["subject"], body["body"],
                          body.get("html", False), body.get("attachments"))
    except Exception as e:
        return {"success": False, "error": str(e)}


# ===== Database Tools =====
@app.post("/tools/db_query")
async def db_query_endpoint(request: Request):
    try:
        body = await request.json()
        return db_query(body.get("db_type", "sqlite"), body.get("connection_string"),
                       body["query"], body.get("limit", 100))
    except Exception as e:
        return {"success": False, "error": str(e)}


# ===== Scheduler Tools =====
@app.post("/tools/add_task")
async def add_task_endpoint(request: Request):
    try:
        body = await request.json()
        return add_task(body["name"], body["command"], body.get("schedule_type", "cron"),
                       body.get("schedule_expr", "* * * * *"))
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/tools/list_tasks")
async def list_tasks_endpoint():
    return list_tasks()


@app.post("/tools/remove_task")
async def remove_task_endpoint(request: Request):
    try:
        body = await request.json()
        return remove_task(body["name"])
    except Exception as e:
        return {"success": False, "error": str(e)}


# ===== Monitoring Tools =====
@app.get("/tools/system_info")
async def system_info_endpoint():
    return system_info()


@app.post("/tools/process_list")
async def process_list_endpoint(request: Request):
    try:
        body = await request.json()
        return process_list(body.get("filter"))
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/tools/process_kill")
async def process_kill_endpoint(request: Request):
    try:
        body = await request.json()
        return process_kill(body["pid"], body.get("signal", 9))
    except Exception as e:
        return {"success": False, "error": str(e)}




# ============================================================================
# Interactive Terminal & Browser Automation
# ============================================================================
# ============================================================================
# WebSocket Interactive Terminal
# ============================================================================

class TerminalSession:
    """Manage a PTY-based terminal session."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.proc = None
        self.pty_fd = None
        self.connected = False

    def start(self):
        """Start a PTY shell."""
        import pty
        pid, self.pty_fd = pty.openpty()
        env = os.environ.copy()
        env['PS1'] = ''
        env['PROMPT_COMMAND'] = ''
        self.proc = subprocess.Popen(
            ['bash', '-i'],
            stdin=self.pty_fd,
            stdout=self.pty_fd,
            stderr=self.pty_fd,
            env=env,
            preexec_fn=os.setsid
        )
        os.close(self.pty_fd)
        self.pty_fd = None
        self.connected = True

    def send(self, data: str) -> str:
        """Send input to the terminal."""
        if not self.connected or not self.proc or self.proc.poll() is not None:
            return "Terminal closed"
        try:
            import os
            if hasattr(self.proc, 'stdin') and self.proc.stdin:
                os.write(self.proc.stdin.fileno(), (data + '\n').encode())
            time.sleep(0.5)
            output = self._read_output()
            return output
        except Exception as e:
            return f"Write error: {str(e)}"

    def _read_output(self) -> str:
        """Read all available output from PTY."""
        import select
        import fcntl
        import termios
        output = ""
        try:
            if self.proc and not self.proc.stdout.closed:
                flags = fcntl.fcntl(self.proc.stdout, fcntl.F_GETFL)
                fcntl.fcntl(self.proc.stdout, fcntl.F_SETFL, flags | os.O_NONBLOCK)
                try:
                    raw = self.proc.stdout.read()
                    if isinstance(raw, bytes):
                        raw = raw.decode('utf-8', errors='replace')
                    output = raw
                except BlockingIOError:
                    pass
                fcntl.fcntl(self.proc.stdout, fcntl.F_SETFL, flags)
        except Exception:
            pass
        return output

    def resize(self, rows: int = 24, cols: int = 80):
        """Resize terminal."""
        try:
            import struct
            import fcntl
            import termios
            import array
            winsize = array.array('h', [rows, cols, 0, 0])
            fcntl.ioctl(self.proc.stdin, termios.TIOCSWINSZ, winsize)
        except Exception:
            pass

    def kill(self):
        """Terminate the terminal session."""
        if self.proc:
            try:
                os.killpg(os.getpgid(self.proc.pid), 9)
            except Exception:
                pass
            self.proc = None
        self.connected = False


# Global session store
terminal_sessions: Dict[str, TerminalSession] = {}


@app.websocket("/ws/terminal")
async def websocket_terminal(websocket: WebSocket):
    """Interactive terminal via WebSocket."""
    session_id = str(uuid.uuid4())[:8]
    await websocket.accept()
    
    session = TerminalSession(session_id)
    session.start()
    terminal_sessions[session_id] = session
    
    try:
        # Send welcome message
        await websocket.send_text(f"Terminal session started (ID: {session_id})")
        await websocket.send_text(f"Type commands. Supported: resize, kill, exec <cmd>")
        
        while True:
            try:
                data = await websocket.receive_text()
                data = data.strip()
                
                if data.startswith("resize "):
                    parts = data.split()
                    if len(parts) >= 3:
                        session.resize(int(parts[1]), int(parts[2]))
                        await websocket.send_text(f"Terminal resized to {parts[1]}x{parts[2]}")
                    continue
                
                if data == "kill":
                    session.kill()
                    del terminal_sessions[session_id]
                    await websocket.send_text("Terminal killed")
                    break
                
                if data.startswith("exec "):
                    cmd = data[5:]
                    output = session.send(cmd)
                    await websocket.send_text(f"$ {cmd}\n{output}")
                else:
                    output = session.send(data)
                    await websocket.send_text(output)
            except WebSocketDisconnect:
                break
    except Exception as e:
        await websocket.send_text(f"Error: {e}")
    finally:
        session.kill()
        terminal_sessions.pop(session_id, None)


# ============================================================================
# Browser Automation with Playwright
# ============================================================================

class BrowserManager:
    """Manage browser instances via Playwright."""
    def __init__(self):
        self.browser = None
        self.context = None
        self.page = None
        self.sessions: Dict[str, Dict[str, Any]] = {}

    async def start(self):
        """Start Playwright browser."""
        from playwright.async_api import async_playwright
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )

    async def stop(self):
        """Close browser."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def start_session(self, session_id: str):
        """Start a new browser session."""
        if not self.context:
            return {"error": "Browser not started. Use /browser/start first."}
        
        page = await self.context.new_page()
        self.sessions[session_id] = {
            "page": page,
            "url": None,
            "started_at": datetime.now().isoformat()
        }
        return {"session_id": session_id, "status": "started"}

    async def navigate(self, session_id: str, url: str) -> Dict[str, Any]:
        """Navigate to a URL."""
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        page = self.sessions[session_id]["page"]
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            title = await page.title()
            self.sessions[session_id]["url"] = url
            
            # Get page info
            height = await page.evaluate("document.body.scrollHeight")
            
            return {
                "success": True,
                "session_id": session_id,
                "url": url,
                "title": title,
                "page_height": height,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def screenshot(self, session_id: str, full_page: bool = False) -> Dict[str, Any]:
        """Take screenshot."""
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        page = self.sessions[session_id]["page"]
        try:
            screenshot_bytes = await page.screenshot(full_page=full_page, type="png")
            import base64
            b64 = base64.b64encode(screenshot_bytes).decode()
            return {
                "success": True,
                "session_id": session_id,
                "image_base64": b64,
                "full_page": full_page,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_text(self, session_id: str, selector: str = None) -> Dict[str, Any]:
        """Get text content."""
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        page = self.sessions[session_id]["page"]
        try:
            if selector:
                elements = await page.query_selector_all(selector)
                result = []
                for el in elements[:50]:
                    text = await el.inner_text()
                    if text.strip():
                        result.append({"selector": selector, "text": text.strip()})
                return {"success": True, "selector": selector, "results": result}
            else:
                text = await page.evaluate("document.body.innerText")
                return {"success": True, "text": text[:10000]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def click(self, session_id: str, selector: str) -> Dict[str, Any]:
        """Click an element."""
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        page = self.sessions[session_id]["page"]
        try:
            await page.click(selector)
            title = await page.title()
            return {
                "success": True,
                "session_id": session_id,
                "title": title,
                "action": f"clicked '{selector}'",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def fill(self, session_id: str, selector: str, value: str) -> Dict[str, Any]:
        """Fill a form field."""
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        page = self.sessions[session_id]["page"]
        try:
            await page.fill(selector, value)
            return {"success": True, "session_id": session_id,
                    "action": f"filled '{selector}' with '{value}'"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def wait_for_selector(self, session_id: str, selector: str, timeout: int = 10000) -> Dict[str, Any]:
        """Wait for an element to appear."""
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        page = self.sessions[session_id]["page"]
        try:
            await page.wait_for_selector(selector, timeout=timeout)
            return {"success": True, "selector": selector, "found": True}
        except Exception:
            return {"success": False, "error": "Selector not found", "selector": selector}

    async def list_sessions(self) -> Dict[str, Any]:
        """List all active browser sessions."""
        return {
            "success": True,
            "sessions": {
                sid: {"url": s["url"], "started_at": s["started_at"]}
                for sid, s in self.sessions.items()
            },
            "count": len(self.sessions)
        }

    async def close_session(self, session_id: str) -> Dict[str, Any]:
        """Close a browser session."""
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        await self.sessions[session_id]["page"].close()
        del self.sessions[session_id]
        return {"success": True, "message": f"Session {session_id} closed"}


# Global browser manager
browser_manager = BrowserManager()




# ============================================================================
# Browser REST Endpoints
# ============================================================================

@app.post("/browser/start")
async def browser_start_endpoint():
    """Start a browser session."""
    session_id = str(uuid.uuid4())[:8]
    result = await browser_manager.start_session(session_id)
    return result


@app.post("/browser/navigate")
async def browser_navigate_endpoint(request: Request):
    """Navigate to URL."""
    try:
        body = await request.json()
        return await browser_manager.navigate(body["session_id"], body["url"])
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/browser/screenshot")
async def browser_screenshot_endpoint(request: Request):
    """Take screenshot."""
    try:
        body = await request.json()
        return await browser_manager.screenshot(body["session_id"], body.get("full_page", False))
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/browser/get_text")
async def browser_get_text_endpoint(request: Request):
    """Get page text."""
    try:
        body = await request.json()
        return await browser_manager.get_text(body["session_id"], body.get("selector"))
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/browser/click")
async def browser_click_endpoint(request: Request):
    """Click element."""
    try:
        body = await request.json()
        return await browser_manager.click(body["session_id"], body["selector"])
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/browser/fill")
async def browser_fill_endpoint(request: Request):
    """Fill form field."""
    try:
        body = await request.json()
        return await browser_manager.fill(body["session_id"], body["selector"], body["value"])
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/browser/wait_for_selector")
async def browser_wait_selector_endpoint(request: Request):
    """Wait for element."""
    try:
        body = await request.json()
        return await browser_manager.wait_for_selector(
            body["session_id"], body["selector"], body.get("timeout", 10000))
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/browser/list_sessions")
async def browser_list_sessions_endpoint():
    """List browser sessions."""
    return await browser_manager.list_sessions()


@app.post("/browser/close_session")
async def browser_close_session_endpoint(request: Request):
    """Close browser session."""
    try:
        body = await request.json()
        return await browser_manager.close_session(body["session_id"])
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# Terminal REST Endpoint (non-WS fallback)
# ============================================================================

@app.post("/terminal/exec")
async def terminal_exec_endpoint(request: Request):
    """Execute command in terminal (non-WebSocket)."""
    try:
        body = await request.json()
        session_id = body.get("session_id")
        
        # Auto-create session if not exists
        if not session_id or session_id not in terminal_sessions:
            session_id = str(uuid.uuid4())[:8]
            terminal_sessions[session_id] = TerminalSession(session_id)
            terminal_sessions[session_id].start()
        
        cmd = body["command"]
        output = terminal_sessions[session_id].send(cmd)
        return {
            "success": True,
            "session_id": session_id,
            "command": cmd,
            "output": output[:5000],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/terminal/list")
async def terminal_list_endpoint():
    """List active terminal sessions."""
    return {
        "success": True,
        "sessions": list(terminal_sessions.keys()),
        "count": len(terminal_sessions)
    }


@app.post("/terminal/kill")
async def terminal_kill_endpoint(request: Request):
    """Kill terminal session."""
    try:
        body = await request.json()
        sid = body["session_id"]
        if sid in terminal_sessions:
            terminal_sessions[sid].kill()
            del terminal_sessions[sid]
            return {"success": True, "message": f"Session {sid} killed"}
        return {"success": False, "error": f"Session {sid} not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# Auto-start browser on server startup
# ============================================================================

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    """Server lifespan with browser management."""
    try:
        await browser_manager.start()
        logger.info("Browser (Chromium) started")
    except Exception as e:
        logger.error(f"Browser start failed: {e}")
    yield
    try:
        await browser_manager.stop()
        logger.info("Browser stopped")
    except Exception as e:
        logger.error(f"Browser stop failed: {e}")

app.router.lifespan_context = lifespan


if __name__ == "__main__":
    logger.info(f"Starting {SERVER_NAME} v{SERVER_VERSION}")
    logger.info(f"Server listening on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")