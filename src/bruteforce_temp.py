# curious as to ho secure your pswd realy is?

import time
from itertools import product

# List of top 50 most common passwords
common_passwords = [
    "123456", "password", "12345678", "qwerty", "123456789", "12345", "1234", "111111",
    "1234567", "dragon", "123123", "baseball", "abc123", "football", "monkey", "letmein",
    "696969", "shadow", "master", "666666", "qwertyuiop", "123321", "mustang", "1234567890",
    "michael", "654321", "superman", "1qaz2wsx", "7777777", "121212", "000000", "qazwsx",
    "123qwe", "killer", "trustno1", "jordan", "jennifer", "zxcvbnm", "asdfgh", "hunter",
    "buster", "soccer", "harley", "batman", "andrew", "tigger", "sunshine", "iloveyou", "2000"
]

def gen_pswd_combos(guess, max_lngt):
    
    unknwn_lngt = max_lngt - len(guess)
    digits = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}|;:,.<>?/~`'
    print(f"unknown length: {unknwn_lngt},")

    for combos in product(digits, repeat=unknwn_lngt):
        yield guess + ''.join(combos)

def is_rl_pswd(pswd, rl_pswd):
    return pswd == rl_pswd

def main():
    max_lngt = int(16)
    guess = ("guess")
    
    print(f"Total length of password: {max_lngt}")
    print(f"guessed unknown: {guess}")
    
    start_time = time.time()  # Start timing

# First, try the common passwords from the wordlist
    for i, common_pswd in enumerate(common_passwords):
        print(f"Trying common password {i}: {common_pswd}")
        if is_rl_pswd(common_pswd, guess):
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            print(f"Correct password found: {common_pswd}. Proceeding to login.")
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
            return
        

   # If not found in common passwords, try the generated combinations
    combos = gen_pswd_combos(guess, max_lngt)
    for i, combo in enumerate(combos):
        print(f"Combo {i}: {combo}\n")
        print(f"Trying pswd: {combo}")
        if is_rl_pswd(combo, guess):
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            print(f"Correct password found: {guess}. Proceeding to login.")
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
            break

if __name__ == "__main__":
    main()







import hashlib

# Original password
password = "password123"

# Encode the password to bytes, then compute SHA-256 hash
hash_object = hashlib.sha256(password.encode('utf-8'))
sha256_hash = hash_object.hexdigest()

print(f"SHA-256 Hash of '{password}': {sha256_hash}")
