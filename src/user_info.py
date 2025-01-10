#!/usr/bin/env python3
import os
import socket

def main():
    # Print the necessary HTTP headers for a CGI script
    print("Content-Type: text/html\r\n\r\n")

    # Gather environment variables in a Pythonic way
    remote_addr = os.environ.get("HTTP_X_FORWARDED_FOR", 
                    os.environ.get("REMOTE_ADDR", "unknown"))
    # Attempt to resolve hostname
    try:
        remote_hostname = socket.gethostbyaddr(remote_addr)[0]
    except Exception:
        remote_hostname = "unknown"

    remote_port          = os.environ.get("REMOTE_PORT",          "unknown")
    user_agent          = os.environ.get("HTTP_USER_AGENT",      "unknown")
    referer_page        = os.environ.get("HTTP_REFERER",         "unknown")
    http_accept         = os.environ.get("HTTP_ACCEPT",          "unknown")
    http_accept_language= os.environ.get("HTTP_ACCEPT_LANGUAGE", "unknown")
    http_accept_encoding= os.environ.get("HTTP_ACCEPT_ENCODING", "unknown")
    http_accept_charset = os.environ.get("HTTP_ACCEPT_CHARSET",  "unknown")
    proxy_x_forwarded   = os.environ.get("HTTP_X_FORWARDED_FOR", "unknown")
    proxy_forwarded     = os.environ.get("HTTP_FORWARDED",       "unknown")
    proxy_client_ip     = os.environ.get("HTTP_CLIENT_IP",       "unknown")
    proxy_via           = os.environ.get("HTTP_VIA",             "unknown")
    proxy_connection    = os.environ.get("HTTP_PROXY_CONNECTION","unknown")

    # Output in HTML format
    print(f"<p><b>Your IP address</b> : {remote_addr}<br>")
    print(f"<b>Your hostname</b> : {remote_hostname}<br>")
    print(f"<b>Your source port</b> : {remote_port}<br></p>")

    print(f"<p><b>Your user agent</b> : {user_agent}<br>")
    print(f"<b>Referer page</b> : {referer_page}<br>")
    print(f"<b>HTTP Accept</b> : {http_accept}<br>")
    print(f"<b>HTTP Accept language</b> : {http_accept_language}<br>")
    print(f"<b>HTTP Accept encoding</b> : {http_accept_encoding}<br>")
    print(f"<b>HTTP Accept charset</b> : {http_accept_charset}<br></p>")

    print(f"<p><b>Proxy XFORWARDED</b> : {proxy_x_forwarded}<br>")
    print(f"<b>Proxy Forwarded</b> : {proxy_forwarded}<br>")
    print(f"<b>Proxy client IP</b> : {proxy_client_ip}<br>")
    print(f"<b>Proxy via</b> : {proxy_via}<br>")
    print(f"<b>Proxy Proxy connection</b> : {proxy_connection}<br></p>")

if __name__ == "__main__":
    main()
