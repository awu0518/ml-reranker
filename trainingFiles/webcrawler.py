import logging
import time
import urllib.request
from threading import Lock, Thread, Event
import socket
from queue import Queue
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse
import json
import trafilatura

# Configs and Global Variables
UA = "azw7225@nyu.edu"
TARGET_TOTAL = 1000
REQUEST_TIMEOUT = 8
THREADS = 10

CONFIG = trafilatura.settings.use_config()
CONFIG.set("DEFAULT", "include_comments", "no")
CONFIG.set("DEFAULT", "include_tables", "no")
CONFIG.set("DEFAULT", "favor_recall", "no")       # more precision, less junk
CONFIG.set("DEFAULT", "min_output_length", "200") # skip super tiny fragments

DISALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".svg", ".ico",
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
    ".odt", ".ods", ".odp", ".rtf", ".txt",
    ".mp3", ".wav", ".aac", ".ogg", ".flac",
    ".mp4", ".avi", ".mov", ".mkv", ".webm",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".iso", ".dmg", ".exe", ".msi", ".bin",
    ".js", ".css", ".json", ".xml", ".csv", ".tsv",
    ".yaml", ".yml"
}

STOP = Event()

logger = logging.getLogger(__name__)
socket.setdefaulttimeout(REQUEST_TIMEOUT)

rpCache: dict[str, RobotFileParser] = {}
rpLock = Lock()

serpQueue = Queue() # holds (JSON, docID)
writeLock = Lock()

# Adds user agent to urllib.request calls
_opener = urllib.request.build_opener()
_opener.addheaders = [("User-Agent", UA)]
urllib.request.install_opener(_opener)

def stripLink(link: str) -> str:
    """Removes any anchors and queries from links"""
    newLink = link.split("#", 1)[0]
    newLink = newLink.split("?", 1)[0]
    return newLink

def validLink(link: str) -> bool:
    """Checks if a link contains cgi or disallowed extension"""
    if not link:
        return False
    if "cgi" in link:
        return False

    lastSegment = urlparse(link).path.rsplit("/", 1)[-1]
    dot = lastSegment.rfind(".")
    if dot != -1 and lastSegment[dot:].lower() in DISALLOWED_EXTENSIONS:
        return False
    
    return True

def canParseRobot(link: str) -> bool:
    """Checks if a link can be fetched per robots.txt"""
    parsed = urlparse(link)
    baseLink = parsed.scheme + "://" + parsed.netloc
    
    with rpLock:
        rp = rpCache.get(baseLink)
        if not rp:
            rp = RobotFileParser()
            robolink = baseLink + "/robots.txt"
            try:
                rp.set_url(robolink)
                rp.read()
            except Exception:
                pass
            rpCache[baseLink] = rp
    try:
        return rp.can_fetch(UA, link)
    except Exception:
        return True

def worker():
    """
    Workers only consume serpQueue and fetch pages.
    """
    while not STOP.is_set():
        jsonObj, docID = serpQueue.get()
        if jsonObj is None:
            break  # global stop

        if not validLink(jsonObj["url"]) or not canParseRobot(jsonObj["url"]):
            continue

        try:
            request = urllib.request.Request(jsonObj["url"], headers={"User-Agent": UA})
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                if response.headers.get_content_type() != "text/html":
                    continue
                charset = response.headers.get_content_charset()
                html = response.read().decode(charset or "utf-8", errors="replace")
        except Exception:
            continue

        outPath = f"queries/{jsonObj['qid']}.jsonl"
        
        text = trafilatura.extract(html, config=CONFIG, url=jsonObj["url"], no_fallback=True)
        if not text:
            continue

        text = " ".join(text.split())

        if len(text) < 200:
            continue

        doc = {
            "query_id": jsonObj["qid"],
            "query": jsonObj["query"],
            "search_rank": jsonObj["rank"],
            "passage": text,
            "passage_id": docID,
            "url": jsonObj["url"]
        }
        with writeLock:
            with open(outPath, "a", encoding="utf-8") as f:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

def main():
    docID = 1
    with open("../links/serp_links.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            serpQueue.put((obj, docID))
            docID += 1

    for _ in range(THREADS):
        serpQueue.put((None, None))

    for _ in range(THREADS):
        Thread(target=worker, daemon=True).start()

    try:
        while not STOP.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        STOP.set()

if __name__ == "__main__":
    main()
