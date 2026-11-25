# -*- coding: utf-8 -*-
"""返回给前端的 PDF 用 AES-256-GCM 加密。

密钥 = SHA-256( UTF-8(session_id + 文件名不含后缀) )，中间不加分隔符。
密文布局：12 字节 nonce + 密文 + 16 字节 tag。
前端用同一把密钥做 AES-GCM 解密后再展示。
"""

from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path

from Crypto.Cipher import AES


def file_stem(file_name: str) -> str:
    return Path(str(file_name or "").strip()).stem


def pdf_aes_key(session_id: str, file_name: str) -> bytes:
    stem = file_stem(file_name)
    return sha256(f"{session_id}{stem}".encode("utf-8")).digest()


def encrypt_pdf(data: bytes, session_id: str, file_name: str) -> tuple[bytes, str]:
    stem = file_stem(file_name)
    nonce = os.urandom(12)
    cipher = AES.new(pdf_aes_key(session_id, file_name), AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return nonce + ciphertext + tag, stem
