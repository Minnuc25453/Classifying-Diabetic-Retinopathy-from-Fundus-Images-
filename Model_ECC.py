import tracemalloc
from tinyec import registry
import secrets
import hashlib
import time


# Function to encrypt a message using ECC
def encrypt_ECC(msg, pubKey):
    # Generate a random private key
    privKey = secrets.randbelow(pubKey.curve.field.n)
    # Calculate the shared secret
    sharedSecret = privKey * pubKey
    sharedSecret = hashlib.sha256(int.to_bytes(sharedSecret.x, 32, 'big')).digest()
    # Encrypt message using XOR
    encryptedMsg = bytes([msg[i] ^ sharedSecret[i % len(sharedSecret)] for i in range(len(msg))])
    return encryptedMsg, privKey * pubKey.curve.g


# Function to decrypt a message using ECC
def decrypt_ECC(encryptedMsg, privKey):
    # Calculate the shared secret
    sharedSecret = privKey * encryptedMsg[1]
    sharedSecret = hashlib.sha256(int.to_bytes(sharedSecret.x, 32, 'big')).digest()
    # Decrypt message using XOR
    decryptedMsg = bytes(
        [encryptedMsg[0][i] ^ sharedSecret[i % len(sharedSecret)] for i in range(len(encryptedMsg[0]))])
    return decryptedMsg


def Model_ECC(data):
    print('ECC')
    tracemalloc.start()  # Start tracking memory usage
    Curve = registry.get_curve('brainpoolP256r1')

    # Generate private and public keys
    privKey = secrets.randbelow(Curve.field.n)
    pubKey = privKey * Curve.g

    # Convert data to bytes
    plain_text = bytes(str(data), 'utf-8')

    # Encryption
    enc_start = time.time()
    encryptedMsg = encrypt_ECC(plain_text, pubKey)
    enc_time = time.time() - enc_start

    # Decryption
    dec_start = time.time()
    decryptedMsg = decrypt_ECC(encryptedMsg, privKey)
    dec_time = time.time() - dec_start

    current, peak = tracemalloc.get_traced_memory()  # Get memory usage
    tracemalloc.stop()  # Stop memory tracking
    Eval = [peak, current, enc_time, dec_time]
    return Eval, encryptedMsg, decryptedMsg
