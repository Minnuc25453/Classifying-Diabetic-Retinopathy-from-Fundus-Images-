from Crypto.Cipher import DES
from Crypto.Util.Padding import pad, unpad
import binascii
from Crypto.Random import get_random_bytes
import tracemalloc
import time

def des_encrypt(plaintext, key):
    ct = time.time()
    # Ensure the key is exactly 8 bytes long (no need to encode since key is already bytes)
    key = key.ljust(8)[:8]

    # Create a DES cipher object
    cipher = DES.new(key, DES.MODE_ECB)

    # Pad the plaintext to be a multiple of DES block size (8 bytes)
    padded_plaintext = pad(plaintext.encode('utf-8'), DES.block_size)

    # Encrypt the plaintext
    ciphertext = cipher.encrypt(padded_plaintext)
    ct = time.time() - ct
    return binascii.hexlify(ciphertext).decode('utf-8'), ct


def des_decrypt(ciphertext, key):
    ct = time.time()
    # Ensure the key is exactly 8 bytes long (no need to encode since key is already bytes)
    key = key.ljust(8)[:8]

    # Create a DES cipher object
    cipher = DES.new(key, DES.MODE_ECB)

    # Convert ciphertext from hex to bytes
    ciphertext = binascii.unhexlify(ciphertext)

    # Decrypt the ciphertext
    padded_plaintext = cipher.decrypt(ciphertext)

    # Unpad the plaintext
    plaintext = unpad(padded_plaintext, DES.block_size).decode('utf-8')
    ct = time.time() - ct

    return plaintext, ct


def Model_DES(data):
    plaintext = str(data)
    key = get_random_bytes(8)  # Generate a random 64-bit key (8 bytes)
    tracemalloc.start()  # Start memory tracking
    ct = time.time()
    ciphertext, Etime = des_encrypt(plaintext, key)
    decrypted_plaintext, Dtime = des_decrypt(ciphertext, key)
    current, peak = tracemalloc.get_traced_memory()  # Get memory usage
    tracemalloc.stop()  # Stop memory tracking
    ct = time.time() - ct
    Eval = [Etime, Dtime, ct, current]
    return Eval
