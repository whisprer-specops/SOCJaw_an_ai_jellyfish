# wofl's brute forcing and padbusting password function.
# eat it bitch.

import itertools
import time
from datetime import datetime
from sympy import isprime, nextprime
import hashlib


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



    # Original password
    password = "password"

    # Encode the password to bytes, then compute SHA-256 hash
    hash_object = hashlib.sha256(password.encode('utf-8'))
    sha256_hash = hash_object.hexdigest()

    print(f"SHA-256 Hash of '{password}': {sha256_hash}")



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


    # Then, try using PadBuster to crack the password cleverly
    for i, padbusted_word in enumerate(raw_hash)
        # Use PadBuster to try cracking the raw hash via Oracle
        raw_hash = hashed(hash, max_lngt)


# ***** ***** ***** ***** ***** 
# 
# insert PadBuster-Oracle code
# 
# ***** ***** ***** ***** ***** 



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

# Main execution
if __name__ == "__main__":
    main_password_check()  # Call the new password-checking function

