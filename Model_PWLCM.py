import time
import tracemalloc
import numpy as np
from Crypto.Random import get_random_bytes


def pwlcm_sequence(x0, μ, length):
    """
    Generate a PWLCM sequence.
    x0: Initial condition (0 < x0 < 1)
    μ: Parameter for PWLCM (0 < μ < 1)
    length: Length of sequence
    """
    sequence = []
    x = x0
    for _ in range(length):
        if x < μ:
            x = x / μ
        else:
            x = (1 - x) / (1 - μ)
        sequence.append(x)
    return np.array(sequence)


def encrypt(plain_text, x0, μ):
    """
    Encrypts the plain text using PWLCM.
    plain_text: Input string
    x0: Initial condition for PWLCM
    μ: Parameter for PWLCM
    """
    ct = time.time()
    # Convert the plain text to bytes
    byte_array = np.frombuffer(plain_text.encode(), dtype=np.uint8)

    # Generate a chaotic sequence of same length as plain_text
    chaotic_sequence = pwlcm_sequence(x0, μ, len(byte_array))

    # Scale chaotic sequence to byte range (0-255)
    chaotic_bytes = np.floor(chaotic_sequence * 256).astype(np.uint8)

    # XOR each byte of the plain text with chaotic sequence
    cipher_text = np.bitwise_xor(byte_array, chaotic_bytes)

    ct = time.time() - ct
    return cipher_text, ct


def decrypt(cipher_text, x0, μ):
    """
    Decrypts the cipher text using PWLCM.
    cipher_text: Encrypted byte array
    x0: Initial condition for PWLCM
    μ: Parameter for PWLCM
    """
    ct = time.time()
    # Generate a chaotic sequence of same length as cipher_text
    chaotic_sequence = pwlcm_sequence(x0, μ, len(cipher_text))

    # Scale chaotic sequence to byte range (0-255)
    chaotic_bytes = np.floor(chaotic_sequence * 256).astype(np.uint8)

    # XOR each byte of the cipher text with chaotic sequence
    decrypted_bytes = np.bitwise_xor(cipher_text, chaotic_bytes)

    # Convert the bytes back to string
    decrypted_text = decrypted_bytes.tobytes().decode()

    ct = time.time() - ct
    return decrypted_text, ct


def Model_PLCM(data, sol=None):
    if sol is None:
        sol = [1]
    plaintext = str(data)
    x0 = 0.7  # Initial condition for PWLCM
    μ = sol[0] #0.5
    key = get_random_bytes(32)  # 256-bit key
    tracemalloc.start()  # Start memory tracking
    ct = time.time()
    # Encrypt the plain text
    cipher_text, Etime = encrypt(plaintext, x0, μ)
    # Decrypt the cipher text
    decrypted_text, Dtime = decrypt(cipher_text, x0, μ)
    current, peak = tracemalloc.get_traced_memory()  # Get memory usage
    tracemalloc.stop()  # Stop memory tracking
    ct = time.time() - ct
    Eval = [Etime, Dtime, ct, current]
    return Eval,cipher_text,decrypted_text
