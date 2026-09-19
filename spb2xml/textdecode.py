"""
Python port of spb2xml's TextDecode.cs -- see NOTICE.md for attribution.

MSFS .spb files store TEXT/MLTEXT property values through a simple
position-dependent substitution cipher rather than plain UTF-8: for a
string of length N, encoded_byte[i] = S[ord(char[i])][i % 250], and an
extra terminator byte encoded_byte[N] = S[0][N % 250] is appended.

S is a fixed table (ripped from spb2xml's TextDecode.Data.cs, which in
turn ripped it from the MSFS binary/tools -- it's a static substitution
table, not a secret/key, so this port is exact).
"""
from textdecode_data import S_TABLE

_COLS = 250

# Build reverse lookup: K[byte_value][col] -> original char code (or None)
_K = [[None] * _COLS for _ in range(256)]
for char_code, row in S_TABLE.items():
    for col in range(_COLS):
        _K[row[col]][col] = char_code


def decode(encoded: bytes) -> str:
    """Mirrors TextDecode.Decode(byte[] encoded)."""
    n = len(encoded) - 1
    if n < 0:
        raise ValueError("encoded data too short")
    chars = []
    for i in range(n):
        col = i % _COLS
        code = _K[encoded[i]][col]
        if code is None:
            raise ValueError(f"no mapping for byte 0x{encoded[i]:02x} at col {col}")
        chars.append(chr(code))
    last_col = n % _COLS
    last = _K[encoded[n]][last_col]
    if last != 0:
        raise ValueError("Unexpected: bad terminator byte")
    return "".join(chars)


def encode(s: str) -> bytes:
    """Mirrors TextDecode.Encode(string str), for self-testing round-trips."""
    out = bytearray(len(s) + 1)
    i = 0
    for i, ch in enumerate(s):
        row = S_TABLE.get(ord(ch))
        out[i] = row[i % _COLS] if row is not None else 0xFF
        i += 0
    out[len(s)] = S_TABLE[0][len(s) % _COLS]
    return bytes(out)


if __name__ == "__main__":
    # self-test: round-trip a handful of strings through encode/decode
    tests = ["", "A", "Hello", "LHBP_B_1_7", "SimPropContainer test 123!", "x" * 500]
    for t in tests:
        enc = encode(t)
        dec = decode(enc)
        status = "OK" if dec == t else "MISMATCH"
        print(f"{status}: {t!r} -> decode(encode(x)) = {dec!r}")
