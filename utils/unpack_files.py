import gzip
import shutil

source_filepath = 'data_raw/NGS-166.00.txt.gz'
dest_filepath = 'data/NGS-166.00.txt'

with gzip.open(source_filepath, 'rb') as f_in:
    with open(dest_filepath, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
