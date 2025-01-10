#!/usr/bin/env python3
"""
A simple Python HTTP proxy that observes and controls HTTP connections,
inspired by the Perl script from megapanzer.com. Demonstrates:

- Listening on a port
- Forking/Threading per connection (here we use threads)
- Blocking certain hosts, URIs, data sizes
- Logging interesting requests
- Logging to a MySQL database
"""

import socket
import threading
import re
import time
import requests
import mysql.connector
import logging

# ---------------------------
# Configuration & Globals
# ---------------------------

LISTEN_HOST = '0.0.0.0'
LISTEN_PORT = 8000
MAX_LISTEN = 30          # Similar to Listen => 30 in the Perl code
MAX_REQUEST_TIME = 20    # seconds
DEBUG_LEVEL = 2

# Database credentials
DB_NAME = "Megapanzer"
DB_USERNAME = "mega"
DB_PASSWORD = "yourPanzerPW"
DB_HOST = "localhost"  # or wherever your MySQL is
DB_TABLE_NAME = "Connection"

# Where to redirect blocked requests
ERROR_URL = "http://www.megapanzer.com/proxy/test.html"
REDIRECT_ON_ERROR = True

# Regexes
BLOCKED_PORTS = {
    # The Perl code *allows* ports 80 and 7000, so we invert that logic here.
    # We'll block anything not in this set.
    80: True,
    7000: True
}

BLOCK_DESTINATIONS = [
    r'\.rapidshare\.com',
    r'\.blogspot\.com',
    r'www\.google\.',
    r'\.yahoo\.',
    r'\.live\.',
    r'\.gov'
]

BLOCK_URI = [
    r'viagra',
    r'cialis',
    r'reply',
    r'gbook',
    r'guestbook',
    r'iframe',
    r'\/\/ads?\.',
    r'http:\/\/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
    r'comment'
]

# Interesting request regex checks
GET_REGEX = r'(pass=|passwd=|pwd=|password=)'
POST_REGEX = GET_REGEX
COOKIE_REGEX = r'(pass=|passwd=|pwd=|password=|session|sid)'
DST_REGEX = r'\.(facebook|linkedin|skype|xing|myspqce|amazon|ebay|hi5|flickr|youtube|craiglist|skyrock|blogger|wordpress|netlog)\.'

# Logging config
LOG_FILE = "./proxy.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG if DEBUG_LEVEL > 0 else logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ---------------------------
# Database Functions
# ---------------------------

def connectDB():
    """
    Connect to the DB and return the connection handle
    """
    try:
        conn = mysql.connector.connect(
            user=DB_USERNAME,
            password=DB_PASSWORD,
            host=DB_HOST,
            database=DB_NAME
        )
        return conn
    except mysql.connector.Error as err:
        print(f"connectDB(): unable to connect: {err}")
        return None


def disconnectDB(conn):
    """Close DB connection."""
    if conn is not None:
        conn.close()


def newConnRec(conn, action, src_ip, dst_ip, dst_port, dst_host, referer,
               uri, req_method, req_size, cookies):
    """
    Insert a new record into the Connection table.
    This is the rough Python version of 'newConnRec' from the Perl script.
    """
    if conn is None:
        return

    query = f"""
        INSERT INTO {DB_TABLE_NAME} 
        (Action, SrcIP, DstIP, DstPort, DstHostName, Referer, URI, 
         RequestMethod, RequestSize, Cookies)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    args = (action, src_ip, dst_ip, dst_port, dst_host, referer, uri,
            req_method, req_size, cookies)
    try:
        cursor = conn.cursor()
        cursor.execute(query, args)
        conn.commit()
        cursor.close()
    except mysql.connector.Error as err:
        print(f"newConnRec: Error: {err}")
        # Attempt reconnect?
        conn.rollback()


# ---------------------------
# Utility Functions
# ---------------------------

def logger(msg, exit_status=0):
    """
    Logger function emulating the Perl 'logger' sub.
    Writes to both the console and the log file, 
    and can exit the process if exit_status != 0.
    """
    # Write to log file via Python's logging
    if exit_status != 0:
        logging.error(msg)
    else:
        logging.info(msg)

    # Also print to console
    print(f"{time.ctime()} - {msg}")

    if exit_status:
        exit(exit_status)


# ---------------------------
# HTTP Proxy Handling
# ---------------------------

def is_blocked_port(port):
    """Return True if this port is NOT allowed."""
    return not BLOCKED_PORTS.get(port, False)

def is_blocked_destination(dst_ip):
    """Check if the destination matches any known blocked domain regex."""
    for pattern in BLOCK_DESTINATIONS:
        if re.search(pattern, dst_ip, re.IGNORECASE):
            return True
    return False

def is_blocked_uri(uri):
    """Check if the URI matches any known blocked substring."""
    for pattern in BLOCK_URI:
        if re.search(pattern, uri, re.IGNORECASE):
            return True
    return False

def handle_client(conn, addr):
    """
    Handle the client connection (in a new thread).
    Emulates the child process logic of the Perl script.
    """
    db_conn = connectDB()
    client_ip = addr[0]
    conn.settimeout(MAX_REQUEST_TIME)

    try:
        # The simplest approach: read a single HTTP request from the client.
        # For multi-request keep-alive, you’d need a loop here.
        request_data = conn.recv(65535).decode('latin-1', errors='replace')
        if not request_data:
            return

        # -----------------------------------
        # Parse the request line
        # -----------------------------------
        # E.g., "GET http://www.example.com/index.html HTTP/1.1"
        request_lines = request_data.split("\r\n")
        if not request_lines:
            return

        request_line = request_lines[0]
        # Very naive parse
        # format: METHOD SP PATH SP HTTP/VERSION
        parts = request_line.split()
        if len(parts) < 3:
            logger(f"[ BLOCK method ] {client_ip} -> invalid request line: {request_line}")
            conn.close()
            return

        method, path, _ = parts
        headers = request_lines[1:]  # the rest is header lines, then blank, then body

        # Grab host header if it exists
        host_header = ""
        referer_header = ""
        cookie_header = ""
        content_length = 0
        post_data = ""

        # We'll track them as we go
        for line in headers:
            if line.lower().startswith("host:"):
                host_header = line.split(":", 1)[1].strip()
            elif line.lower().startswith("referer:"):
                referer_header = line.split(":", 1)[1].strip()
            elif line.lower().startswith("cookie:"):
                cookie_header = line.split(":", 1)[1].strip()
            elif line.lower().startswith("content-length:"):
                cl_str = line.split(":", 1)[1].strip()
                if cl_str.isdigit():
                    content_length = int(cl_str)

        # If there's a body, get it
        # (only if method is POST or PUT, typically)
        # We read out of the socket the needed content_length
        if content_length > 0:
            # The request_data may already contain the body if small enough,
            # otherwise, we might need to read more from the socket.
            # For brevity, let's assume it's fully in `request_data`
            # after the double CRLF
            split_index = request_data.find("\r\n\r\n")
            if split_index != -1:
                post_data = request_data[split_index+4:]
                # If the post_data is shorter than content_length,
                # we’d read more. Let's skip that for brevity.

        # Resolve host and port
        # If path is like "http://host:port/something", parse out.
        # Otherwise assume port 80
        dst_port = 80
        dst_ip = ""
        # If path starts with "http://..."
        if path.lower().startswith("http"):
            # Attempt a naive parse
            # For robust usage, consider using urllib.parse
            match = re.match(r"https?://([^/]+)(.*)", path, re.IGNORECASE)
            if match:
                host_part = match.group(1)
                uri_part = match.group(2)
                # Check if there's a port
                if ":" in host_part:
                    host_only, port_only = host_part.split(":", 1)
                    dst_ip = host_only
                    try:
                        dst_port = int(port_only)
                    except ValueError:
                        dst_port = 80
                else:
                    dst_ip = host_part
                full_uri = "/" + uri_part.lstrip("/")
            else:
                # fallback
                dst_ip = host_header
                full_uri = path
        else:
            # Possibly relative request
            dst_ip = host_header
            full_uri = path

        if not dst_ip:
            # If we still have no host, we can’t proceed
            logger(f"[ BLOCK address ] {client_ip} -> no valid host", 0)
            conn.close()
            return

        # Perform the block checks
        # 1. Port check
        if is_blocked_port(dst_port):
            if REDIRECT_ON_ERROR:
                send_redirect(conn, ERROR_URL)
            logger(f"[ BLOCK port ] {client_ip} -> {dst_ip}:{dst_port}")
            conn.close()
            return

        # 2. Host block check
        if is_blocked_destination(dst_ip):
            if REDIRECT_ON_ERROR:
                send_redirect(conn, ERROR_URL)
            logger(f"[ BLOCK address ] {client_ip} -> {dst_ip}")
            conn.close()
            return

        # 3. URI block check
        if is_blocked_uri(full_uri):
            if REDIRECT_ON_ERROR:
                send_redirect(conn, ERROR_URL)
            logger(f"[ BLOCK URI ] {client_ip} -> {dst_ip} -> {full_uri}")
            conn.close()
            return

        # 4. Content-length block
        if content_length > 8192:
            if REDIRECT_ON_ERROR:
                send_redirect(conn, ERROR_URL)
            logger(f"[ BLOCK data size ] {client_ip} -> {dst_ip} : Content-Length: {content_length}")
            conn.close()
            return

        # 5. Method block check
        # The Perl code tries to block if method is not GET, POST, HEAD.
        # But the if statement in the Perl is a bit odd.
        allowed_methods = ["GET", "POST", "HEAD"]
        if method not in allowed_methods:
            logger(f"[ BLOCK method ] {client_ip} -> {dst_ip} : method {method}")
            conn.close()
            return

        # 6. Interesting request logging check
        is_interesting = False
        auth_header = False

        # Fake parse for 'Authorization'
        for line in headers:
            if line.lower().startswith("authorization:"):
                auth_header = True
                break

        if auth_header:
            is_interesting = True
        if re.search(GET_REGEX, full_uri, re.IGNORECASE):
            is_interesting = True
        if re.search(POST_REGEX, post_data, re.IGNORECASE):
            is_interesting = True
        if re.search(COOKIE_REGEX, cookie_header or "", re.IGNORECASE):
            is_interesting = True

        # Also check if destination is in DST_REGEX
        if is_interesting and re.search(DST_REGEX, dst_ip, re.IGNORECASE):
            # Log to DB
            newConnRec(
                db_conn, "Request",
                client_ip, dst_ip, dst_port, host_header,
                referer_header, full_uri, method, content_length, cookie_header
            )

        # If not blocked, forward the request using requests
        # Build the new request
        forward_headers = {}
        for line in headers:
            if not line.strip():
                continue
            # e.g. "Host: example.com"
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            # remove "Connection" or "Proxy-Connection"
            if k.lower() in ["connection", "proxy-connection"]:
                continue
            forward_headers[k] = v

        # X-Forwarded-For
        forward_headers["X-Forwarded-For"] = client_ip

        # Actually do the request with 'requests'
        # Construct the final URL
        final_url = f"http://{dst_ip}:{dst_port}{full_uri}"

        session = requests.Session()
        response = None
        if method == "GET":
            response = session.get(final_url, headers=forward_headers, data=None, allow_redirects=False)
        elif method == "POST":
            response = session.post(final_url, headers=forward_headers, data=post_data, allow_redirects=False)
        elif method == "HEAD":
            response = session.head(final_url, headers=forward_headers, allow_redirects=False)

        if response is not None:
            # Send the response back to the client
            send_response(conn, response)
            title = "REQUEST W/ COOKIE" if cookie_header else "REQUEST"
            if DEBUG_LEVEL > 1:
                logger(f"[ {title} {is_interesting} ] - {client_ip} -> {method} {full_uri} - {len(response.content)} bytes")
    except socket.timeout:
        logger(f"[ TIMEOUT ] - {client_ip} -> {dst_ip} - {method} {full_uri if 'full_uri' in locals() else ''}")
    except Exception as e:
        logger(f"[ ERROR ] handle_client: {e}")
    finally:
        conn.close()
        disconnectDB(db_conn)

def send_redirect(conn, url):
    """
    Sends an HTTP 302 redirect to the client.
    """
    response_str = (
        "HTTP/1.1 302 Found\r\n"
        f"Location: {url}\r\n"
        "Connection: close\r\n"
        "\r\n"
    )
    conn.sendall(response_str.encode('utf-8'))

def send_response(conn, response):
    """
    Convert a 'requests' response to a raw HTTP response
    and send it back to the client connection.
    """
    # Build status line
    status_line = f"HTTP/1.1 {response.status_code} {response.reason}\r\n"

    # Build headers
    # We only copy a subset to keep it simple
    # e.g. Content-Type, Content-Length, etc.
    headers_str = ""
    for k, v in response.headers.items():
        # Don’t send transfer-encoding with chunked, or content-encoding
        # in a naive manner. But let's keep it simple for now.
        if k.lower() not in ["transfer-encoding", "connection"]:
            headers_str += f"{k}: {v}\r\n"

    body = response.content

    msg = status_line + headers_str + "Connection: close\r\n\r\n"
    conn.sendall(msg.encode('utf-8'))

    if response.request.method != "HEAD":
        conn.sendall(body)

# ---------------------------
# Main Proxy Loop
# ---------------------------

def main():
    # Open a socket and listen
    proxy_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    proxy_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    proxy_server.bind((LISTEN_HOST, LISTEN_PORT))
    proxy_server.listen(MAX_LISTEN)

    logger(f"Proxy listening on {LISTEN_HOST}:{LISTEN_PORT}")

    while True:
        try:
            client_conn, client_addr = proxy_server.accept()
            # Start a new thread to handle the client
            t = threading.Thread(
                target=handle_client,
                args=(client_conn, client_addr),
                daemon=True
            )
            t.start()
        except KeyboardInterrupt:
            logger("Shutting down proxy server...", 0)
            proxy_server.close()
            break

if __name__ == "__main__":
    main()
