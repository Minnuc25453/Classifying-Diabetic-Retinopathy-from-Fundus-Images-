
import tracemalloc
import time
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad


def encrypt_aes(plaintext, key):
    ct = time.time()
    cipher = AES.new(key, AES.MODE_CBC)
    ciphertext = cipher.encrypt(pad(plaintext.encode(), AES.block_size))
    ct = time.time() - ct
    return ciphertext, cipher.iv, ct


def decrypt_aes(ciphertext, key, iv):
    ct = time.time()
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
    ct = time.time() - ct
    return decrypted_plaintext.decode(), ct


def Model_AES(data):
    plaintext = str(data)
    key = get_random_bytes(32)  # 256-bit key
    tracemalloc.start()  # Start memory tracking
    ct = time.time()
    ciphertext, iv, Etime = encrypt_aes(plaintext, key)
    decrypted_plaintext, Dtime = decrypt_aes(ciphertext, key, iv)
    current, peak = tracemalloc.get_traced_memory()  # Get memory usage
    tracemalloc.stop()  # Stop memory tracking
    ct = time.time() - ct
    Eval = [Etime, Dtime, ct, current]
    return Eval



