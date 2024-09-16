import struct

# PNGファイルのチャンクを読み取る関数
def read_png_chunks(filename):
    with open(filename, 'rb') as f:
        # PNGヘッダーを確認する（8バイト）
        png_header = f.read(8)
        if png_header != b'\x89PNG\r\n\x1a\n':
            raise ValueError("これはPNGファイルではありません")
        
        chunks = []
        
        while True:
            # チャンクの長さ（4バイト）
            chunk_length_data = f.read(4)
            if len(chunk_length_data) == 0:
                break  # ファイルの終わりに達した
            
            # チャンクの長さを整数に変換
            chunk_length = struct.unpack('!I', chunk_length_data)[0]
            
            # チャンクタイプ（4バイト）
            chunk_type = f.read(4).decode('ascii')
            
            # チャンクデータ（長さは chunk_length で決定）
            chunk_data = f.read(chunk_length)
            
            # CRC（4バイト）
            chunk_crc = f.read(4)
            
            # チャンクの情報を保存
            chunks.append({
                'type': chunk_type,
                'length': chunk_length,
                'data': chunk_data,
                'crc': chunk_crc
            })
            
            # IENDチャンクを見つけたら終了
            if chunk_type == 'IEND':
                break
        
        return chunks

# チャンクの内容を表示する関数
def print_chunks_info(chunks):
    for chunk in chunks:
        print(f"チャンクタイプ: {chunk['type']}")
        print(f"データ長: {chunk['length']}バイト")
        if chunk['length'] > 0:
            print(f"データの一部: {chunk['data'][:50]}" + ("..." if len(chunk["data"]) >= 50 else ""))  # データの一部を表示
        print(f"CRC: {chunk['crc'].hex()}")
        print('-' * 40)

# PNGファイルを読み取り、チャンク情報を表示する
filename = 'output.png'
chunks = read_png_chunks(filename)
print_chunks_info(chunks)