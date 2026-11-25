# -*- coding: utf-8 -*-
"""PDF 加密。入库和返回给前端用的不是同一把钥匙。

算法是 AES-256-GCM。钥匙是 SHA-256 出来的 32 字节，不是把文件做成哈希就完了，
哈希解不回去。

入库钥匙的原文：store:文件名，文件名不带后缀。
返回钥匙的原文：session_id 和文件名直接连在一起，文件名不带后缀，中间不加符号。
文件布局：7 字节头 PCBENC1，接着 12 字节 nonce，16 字节 tag，后面是密文。
没有这个头的旧 PDF 保持原样。
前端用 WebCrypto 时，要把密文和 tag 按「密文在前、tag 在后」拼起来再解密。
"""

from __future__ import annotations

import hashlib
import secrets
from pathlib import Path

_MAGIC = b"PCBENC1"
_NONCE_LEN = 12
_TAG_LEN = 16


def file_stem(file_name: str) -> str:
    """文件名去掉后缀。路径只取最后一段。"""
    name = str(file_name or "").strip().replace("\\", "/").split("/")[-1]
    return Path(name).stem


def _store_key(file_name: str) -> bytes:
    return hashlib.sha256(("store:" + file_stem(file_name)).encode("utf-8")).digest()


def view_key(session_id: str, file_name: str) -> bytes:
    """前端用同一条规则自己算出钥匙：SHA-256(session_id + 不带后缀的文件名)。"""
    text = str(session_id or "") + file_stem(file_name)
    return hashlib.sha256(text.encode("utf-8")).digest()


def is_encrypted(data: bytes | None) -> bool:
    return bool(data) and bytes(data).startswith(_MAGIC)


def _encrypt(data: bytes, key: bytes) -> bytes:
    from Crypto.Cipher import AES

    nonce = secrets.token_bytes(_NONCE_LEN)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return _MAGIC + nonce + tag + ciphertext


def _decrypt(data: bytes, key: bytes) -> bytes:
    from Crypto.Cipher import AES

    raw = bytes(data)
    nonce = raw[len(_MAGIC) : len(_MAGIC) + _NONCE_LEN]
    tag_at = len(_MAGIC) + _NONCE_LEN
    tag = raw[tag_at : tag_at + _TAG_LEN]
    ciphertext = raw[tag_at + _TAG_LEN :]
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag)


def encrypt_stored(data: bytes, file_name: str) -> bytes:
    """入库前加密。已经加密过的不再包一层。"""
    if not data or is_encrypted(data):
        return data
    return _encrypt(data, _store_key(file_name))


def decrypt_stored(data: bytes, file_name: str) -> bytes:
    """读库之后解密。旧文件没有文件头，原样拿去画图。"""
    if not data or not is_encrypted(data):
        return data
    return _decrypt(bytes(data), _store_key(file_name))


def encrypt_for_view(data: bytes, session_id: str, file_name: str) -> bytes:
    """返回给前端之前加密。钥匙能用 session_id 和文件名重算。"""
    if not data:
        return data
    return _encrypt(data, view_key(session_id, file_name))
