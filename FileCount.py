import os

# ディレクトリのパス
directory = '/Volumes/SSD/img_align_celeba'

# 画像ファイルのリストを取得
image_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.jpg')]

print(f'ファイル数: {len(image_files)}')
