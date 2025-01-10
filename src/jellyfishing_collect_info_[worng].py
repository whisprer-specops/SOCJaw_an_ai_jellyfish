#!/usr/bin/env python3
#
# smol test jellyfishings fella:
# he's just to see if the http parsing code and
# some other bits bobs o navigationy/decidey
# type stuff works ok - nothing particularly
# exciting to see here really...
# p.s. playing during this codings:
# https://www.youtube.com/v/y85YXMBRASc "westbam - hard times"
# thnx to carrumba @megapanzer.com

import requests
import random
import re
from bs4 import BeautifulSoup
import os
import socket


class InternetJellyfish:
    """
    A simulated 'Jellyfish' that can navigate around the internet
    and scrape page data. 
    """

    # URL Configuration
    URL_LIST = [
    "https://dzen.ru/?yredirect=true",
    "https://www.kaggle.com/",
    "https://arxiv.org/",
    "https://scholar.google.com/",
    "https://alltop.com/",
    "https://feedly.com/",
    "https://news.ycombinator.com/",
    "https://github.com/",
    "https://stackexchange.com/",
    "https://dmoztools.net/",
    "https://archive.org/",    
    "https://twitter.com/",
    "http://ajaxian.com",
    "http://digg.com",
    "http://en.wikipedia.org",
    "https://www.wikipedia.org/",
    "http://english.aljazeera.net/HomePage",
    "https://online.wellsfargo.com",
    "http://de.wikipedia.org",
    "http://login.yahoo.com",
    "http://mail.google.com",
    "mail.yahoo.com",
    "http://my.yahoo.com",
    "http://reddit.com",
    "http://seoblackhat.com",
    "http://slashdot.org",
    "http://techfoolery.com",
    "http://weblogs.asp.net/jezell",
    "http://www.amazon.com",
    "http://www.aol.com",
    "http://www.apple.com",
    "http://www.ask.com",
    "http://www.bankofamerica.com",
    "http://www.bankone.com",
    "http://www.bing.com",
    "http://www.blackhat.com",
    "http://www.blogger.com",
    "http://www.bloglines.com",
    "http://www.bofa.com",
    "http://www.capitalone.com",
    "http://www.cenzic.com",
    "http://www.cgisecurity.com",
    "http://www.chase.com",
    "http://www.citibank.com",
    "http://www.comerica.com",
    "http://www.craiglist.org",
    "http://www.e-gold.com",
    "http://www.ebay.com",
    "http://www.etrade.com",
    "http://www.expedia.com",
    "http://www.flickr.com",
    "http://www.go.com",
    "http://www.google.ca",
    "http://www.google.com",
    "http://www.google.com.br",
    "http://www.google.ch",
    "http://www.google.cn",
    "http://www.google.co.jp",
    "http://www.google.de",
    "http://www.google.es",
    "http://www.google.fr",
    "http://www.hattrick.org",
    "http://www.hsbc.com",
    "http://www.hi5.com",
    "http://www.icq.com",
    "http://www.imdb.com",
    "http://www.imageshack.us",
    "http://www.jailbabes.com",
    "http://www.linkedin.com",
    "http://www.tagged.com",
    "http://www.globo.com",
    "http://www.megavideo.com",
    "http://www.kino.to",
    "http://www.google.ru",
    "http://www.friendster.com",
    "http://www.nytimes.com",
    "http://www.megaclick.com",
    "http://www.mininova.org",
    "http://www.thepiratebay.org",
    "http://www.netlog.com",
    "http://www.orkut.com",
    "http://www.megacafe.com",
    "http://www.answers.com",
    "http://www.scribd.com",
    "http://www.xing.com",
    "http://www.partypoker.com",
    "http://www.skype.com",
    "http://mail.google.com",
    "http://www.microsoft.com",
    "http://www.msn.com",
    "http://www.myspace.com",
    "http://www.ntobjectives.com",
    "http://www.passport.net",
    "http://www.paypal.com",
    "http://www.photobucket.com",
    "http://www.pornhub.com",
    "http://www.reddit.com",
    "http://www.redtube.com",
    "http://www.sourceforge.net",
    "http://www.spidynamics.com",
    "http://www.statefarm.com",
    "http://www.twitter.com",
    "http://www.vkontakte.ru",
    "http://www.wachovia.com",
    "http://www.wamu.com",
    "http://www.watchfire.com",
    "http://www.webappsec.org",
    "http://www.wellsfargo.com",
    "http://www.whitehatsec.com",
    "http://www.wordpress.com",
    "http://www.xanga.com",
    "http://www.yahoo.com",
    "http://seoblackhat.com",
    "http://www.alexa.com",
    "http://www.youtube.com",
    "https://banking.wellsfargo.com",
    "https://commerce.blackhat.com",
    ]

    # Define your URL list here
    random_url = random.choice(URL_LIST)
    print(f"Randomly selected URL: {random_url}")

    def __init__(self, random_url):
        # The current position of our Jellyfish in the vast internet
        self.position = random_url
        # A list that stores page data we've “sensed”
        self.data = []

    def navigate(self):
        """
        Simulate "movement" by following a random link on the current page.
        If no links are found, reset position to the default page.
        """
        try:
            response = requests.get(self.position, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                links = soup.find_all('a', href=True)
                if links:
                    next_link = random.choice(links)['href']
                    # Optional: If next_link is relative, join it with the base URL
                    if next_link.startswith('/'):
                        from urllib.parse import urljoin
                        next_link = urljoin(self.position, next_link)
                    self.position = next_link
                else:
                    print("[navigate] No links found on this page. Going back to default.")
                    self.position = "https://commons.wikimedia.org/wiki/Main_Page"
            else:
                print(f"[navigate] HTTP status {response.status_code}. Resetting to default.")
                self.position = "https://commons.wikimedia.org/wiki/Main_Page"
        except requests.RequestException as e:
            print(f"[navigate] Error: {e}")
            # If an error occurs, go back to default
            self.position = "https://commons.wikimedia.org/wiki/Main_Page"

    def sense(self):
        """
        Simulate "sensing" by scraping the current page and appending
        its raw HTML text to our data list.
        """
        try:
            response = requests.get(self.position, timeout=10)
            if response.status_code == 200:
                self.data.append(response.text)
            else:
                print(f"[sense] HTTP status {response.status_code}. Unable to sense page data.")
        except requests.RequestException as e:
            print(f"[sense] Error: {e}")

    def decide(self):
        """
        Decide whether to navigate or sense based on what's found 
        in the latest page data (or any other condition).
        """
        # Make sure we have some data to analyze; if none, default to 'sense'
        if not self.data:
            self.sense()
            return

        # Get the most recent snippet of data
        latest_data = self.data[-1]

        # Example condition: if the data includes "login" or "password",
        # we might want to "sense" more. Otherwise, "navigate".
        if "login|session|sid" in latest_data.lower() or "passwd|pwd|password" in latest_data.lower():
            print("[decide] Found credentials-related text; focusing on sense action.")
            self.sense()
        else:
            print("[decide] No credentials found; continuing to navigate.")
            self.navigate()


    def http_stuff():
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


    def parse_http_stuff_output(http_output):
        """
        A placeholder function that parses the text output from some
        hypothetical PHP script, extracting user info, proxy info,
        hardware info, etc. using regex.

        Returns:
          user_info, proxy_info, hardware_info, networking_info,
          installed_plugins, browser_history
        """

    # Initialize our data structures
    user_info = {}
    proxy_info = {}
    hardware_info = {}
    networking_info = {}
    installed_plugins = []
    browser_history = []

    # Example: Patterns for user info
    user_info_patterns = {
        # Already existing:
        "IP Address": r"Your IP address\s*:\s*(.*)",
        "Hostname": r"Your hostname\s*:\s*(.*)",
        "Source Port": r"Your source port\s*:\s*(.*)",
        "User Agent": r"Your user agent\s*:\s*(.*)",

        # Additional lines from the snippet:
        "Referer Page": r"Referer page\s*:\s*(.*)",
        "HTTP Accept": r"HTTP Accept\s*:\s*(.*)",
        "HTTP Accept Language": r"HTTP Accept language\s*:\s*(.*)",
        "HTTP Accept Encoding": r"HTTP Accept encoding\s*:\s*(.*)",
        "HTTP Accept Charset": r"HTTP Accept charset\s*:\s*(.*)",
    }

    proxy_info_patterns = {
        "Proxy XFORWARDED": r"Proxy XFORWARDED\s*:\s*(.*)",
        "Proxy Forwarded": r"Proxy Forwarded\s*:\s*(.*)",
        "Proxy Client IP": r"Proxy client IP\s*:\s*(.*)",
        "Proxy Via": r"Proxy via\s*:\s*(.*)",
        "Proxy Connection": r"Proxy Proxy connection\s*:\s*(.*)",
    }

    hardware_info_patterns = {
        # Derived from the JavaScript part that writes to the document:
        "Browser Name": r"Browser name\).*:\s*(.*)",
        "Browser Version": r"Browser version\).*:\s*(.*)",
        "Platform": r"Platform\).*:\s*(.*)",
        "History Entries": r"History entries\).*:\s*(.*)",
        # This pattern might need special handling because it includes numbers & text
        # e.g. "Screen size : 1920 x 1080 x 24 bpp (available for browser: 1920 x 1040)"
        "Screen Size": r"Screen size\).*:\s*(.*)",
    }

    networking_info_patterns = {
        # From the JavaScript part about local addresses:
        "Localhost Address": r"Localhost address\).*:\s*(.*)",
        "Local NIC Address": r"Local NIC address\).*:\s*(.*)",
    }

    installed_plugins_patterns = {
        # The script prints "plugin.name (plugin.filename)"
        # We'll capture that pattern. Something like "PluginName (PluginFileName)"
        "Plugins": r"(.*?) \((.*?)\)<br>",
        # Or, you may store them as a single pattern and parse the matches differently
    }

    browser_history_patterns = {
        # The snippet has a “stealHistory method” that writes out visited websites in a loop:
        #   document.write(websites[counter], "<br>\n");
        # So you might just capture lines that look like a hyperlink or text with <br>.
        "Visited Website": r">([^<]+)<br>\n",
    }


    # parse logic for user info, proxy_info, hardware_info, etc.:
    for key, pattern in user_info_patterns.items():
        match = re.search(pattern, php_output, re.IGNORECASE)
        if match:
            # .strip() in case of trailing spaces
            user_info[key] = match.group(1).strip()

    for key, pattern in proxy_info_patterns.items():
        match = re.search(pattern, php_output, re.IGNORECASE)
        if match:
            proxy_info[key] = match.group(1).strip()

    for  key, pattern in hardware_info_patterns.items():
        match = re.search(pattern, php_output, re.IGNORECASE)
        if match:
            hardware_info[key] = match.group(1).strip()

    for  key, pattern in networking_info_patterns.items():
        match = re.search(pattern, php_output, re.IGNORECASE)
        if match:
            networking_info[key] = match.group(1).strip()

    for  key, pattern in installed_plugins_patterns.items():
        match = re.search(pattern, php_output, re.IGNORECASE)
        if match:
            installed_plugins[key] = match.group(1).strip()

    for  key, pattern in browser_history_patterns.items():
        match = re.search(pattern, php_output, re.IGNORECASE)
        if match:
            browser_history[key] = match.group(1).strip()

        return (user_info,
                proxy_info,
                hardware_info,
                networking_info,
                installed_plugins,
                browser_history)


def main():
    """
    Example usage: 
    - Create an InternetJellyfish
    - Let it run for 10 cycles, deciding whether to navigate or sense
    - Print out some of the data each time
    """
    jellyfish = InternetJellyfish()

    for i in range(10):
        jellyfish.decide()
        print(f"[Cycle {i+1}] Current Position:", jellyfish.position)
        if jellyfish.data:
            latest_data = jellyfish.data[-1]
            print("[Cycle {}] Data snippet:".format(i+1), latest_data[:100], "...")
        print("---")

    # If at some point you fetch output from your http_stuff script, parse it:
    # http_stuff_output = "...some text from the server..."
    # user_info, proxy_info, hw_info, net_info, plugins, history = parse_http_stuff_output(http_stuff_output)


if __name__ == "__main__":
    main()
