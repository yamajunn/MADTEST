import struct
import zlib

# PNGヘッダ
png_header = b'\x89PNG\r\n\x1a\n'

# IHDRとIENDを含む既存のPNGデータ
with open("input.png", "rb") as f:
    png_data = f.read()

# abcdチャンクのデータ
custom_data = b'AI-generated image'

# チャンク長
length = struct.pack('!I', len(custom_data))

# チャンクタイプ（abcd）
chunk_type = b'gnAI'

# CRCの計算
crc = struct.pack('!I', zlib.crc32(chunk_type + custom_data) & 0xffffffff)

# チャンクを組み立て
custom_chunk = length + chunk_type + custom_data + crc

# IHDRの後、IENDの前にチャンクを挿入する
with open("output.png", "wb") as f:
    f.write(png_data[:-12] + custom_chunk + png_data[-12:])