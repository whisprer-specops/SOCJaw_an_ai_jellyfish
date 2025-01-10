# modified Jellyfish code to make HTTP requests through the expecc
# route, directly accessing the internet. it uses the requests library
# in Python to make HTTP requests.
#
# run this modified Jellyfish code in one terminal and the
# HTTP proxy server code in another terminal to see the
# Jellyfish navigating through the internet.
#
# see `C:\users\user\desktop\http_proxy_for_jellyfish.py``


import requests
from bs4 import BeautifulSoup
import random
import http.server
import socketserver
import logging


# Set up logging
logging.basicConfig(filename='crawler.log', level=logging.DEBUG)

# log requests that match the following patterns
gGETRegex = "(pass=|passwd=|pwd=|password=)"
gPOSTRegex = gGETRegex
gCookieRegex = "(pass=|passwd=|pwd=|password=|session|sid)"
gDstRegex = "\.(win|windows|cammodels|onlyfans|chaturbate|facebook|linkedin|skype|xing|myspqce|amazon|ebay|hi5|flickr|youtube|craiglist|skyrock|blogger|wordpress|netlog)\."


# general settings
gLogFile = "./AAA.txt"
gDEBUG = 2

gBLOCKDestinations = ['\.rapidshare\.com',  
                      '\.blogspot\.com',
                      'www\.google\.',
                      '\.yahoo\.',
                      '\.live\.',
                      '\.gov',
                      ]

gBLOCKURI = ['viagra',
             'cialis',
             'reply',
             'gbook',
             'guestbook',
             'iframe',
             '\/\/ads?\.',
             'http:\/\/[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}',
             'comment'
             ]


visited_urls = set()

def crawl(url, depth=1, max_depth=3):
    if depth > max_depth:
        return

    try:
        if url in visited_urls:
            print(f"Already visited: {url}")
            return

        visited_urls.add(url)

        response = requests.get(url)
        if response.status_code == 200:
            print(f"Crawling: {url}")

            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=True)

            for link in links:
                next_url = link['href']
                
                # Steer away from URLs ending with ".br"
                if next_url.endswith('.br'):
                    continue
                
                # Steer towards URLs ending with ".com"
                if next_url.endswith('.com'):
                    crawl(next_url, depth + 1, max_depth)

                if next_url.startswith('http'):
                    crawl(next_url, depth + 1, max_depth)
                # You can add more conditions here to steer towards or away from other URLs                
                
    except Exception as e:
        print(f"Error crawling {url}: {e}")

class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.proxy_request()

    def do_POST(self):
        self.proxy_request()

    def proxy_request(self):
        try:
            request_headers = dict(self.headers)
            request_body = None
            if 'Content-Length' in request_headers:
                content_length = int(request_headers['Content-Length'])
                request_body = self.rfile.read(content_length)

            # parse request URL
            parsed_url = urllib.parse.urlparse(self.path)
            dest_host = parsed_url.hostname
            dest_port = parsed_url.port if parsed_url.port else 80
            dest_path = parsed_url.path

            # check if destination is blocked
            for pattern in gBLOCKDestinations:
                if re.search(pattern, dest_host, re.IGNORECASE):
                    self.send_error(HTTPStatus.FORBIDDEN)
                    return

            # Block certain URIs
            for pattern in gBLOCKURI:
                if re.search(pattern, self.path, re.IGNORECASE):
                    self.send_error(HTTPStatus.FORBIDDEN)
                    return

             # Log interesting requests
            is_interesting = 0
            if 'Authorization' in request_headers:
                is_interesting += 1
            if re.search(gGETRegex, self.path, re.IGNORECASE):
                is_interesting += 1
            if request_body and re.search(gPOSTRegex, request_body.decode(), re.IGNORECASE):
                is_interesting += 1
            if 'Cookie' in request_headers and re.search(gCookieRegex, request_headers['Cookie'], re.IGNORECASE):
                is_interesting += 1
            if is_interesting > 0 and re.search(gDstRegex, dest_host, re.IGNORECASE):
                log_request(self.client_address[0], self.command, self.path)

            # Prepare request headers
            del request_headers['Proxy-Connection']
            if 'Connection' in request_headers:
                del request_headers['Connection']
            request_headers['Connection'] = 'Close'
            request_headers['X-Forwarded-For'] = self.client_address[0]

            # Send request to destination server
            connection = http.client.HTTPConnection(dest_host, dest_port)
            connection.request(self.command, dest_path, headers=request_headers, body=request_body)
            response = connection.getresponse()

            # Send response back to client
            self.send_response(response.status)
            for header, value in response.getheaders():
                self.send_header(header, value)
            self.end_headers()
            self.wfile.write(response.read())

        except Exception as e:
            print("Error:", e)
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)


def log_request(client_ip, method, uri):
    try:
        with open(gLogFile, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - REQUEST - {client_ip} -> {method} {uri}\n")
    except Exception as e:
        print("Error logging request:", e)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def signal_handler(sig, frame):
    print('Exiting...')
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, signal_handler)
    server = ThreadedHTTPServer(('0.0.0.0', 8000), ProxyHandler)
    server.serve_forever()



# ***** ***** ***** *****


    # Start the HTTP server
    PORT = 8000
    with socketserver.TCPServer(("", PORT), ProxyHandler) as httpd:
        print(f"HTTP server started on port {PORT}")
        httpd.serve_forever()

    # Start crawling
    crawl(start_url)


# This code will now crawl web pages starting from the specified
# start_url crawls through the links on the provided webpage up
# to a certain depth (max_depth). adjusting the max_depth parameter
# controls how deep the crawler traverses through the links.Each 
# HTTP request made by the requests library that can be followed
# is an option allowing the Jellyfish to navigate the internet
# whilst observing HTTP connections.