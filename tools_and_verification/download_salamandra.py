
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='BSC-LT/salamandra-7b-instruct',
    local_dir=r'C:/Users/rdb104/Documents/caserepos/models',
    ignore_patterns=['*.bin']
)
print('Download complete')
