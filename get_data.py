import sys,os,glob
import pandas as pd

input_dir = sys.argv[1]
output_dir = sys.argv[2]
if not os.path.exists(os.path.join(output_dir, 'svs')):
    os.makedirs(os.path.join(output_dir, 'svs'), exist_ok=True)

dirs = os.listdir(input_dir)
dirs = [d for d in dirs if d[0] != '.']

svs_filenames = []
DX_filenames = []
labels = []
for ind, d in enumerate(dirs):
    filenames = os.listdir(os.path.join(input_dir, d))
    filenames = [filename for filename in filenames if filename[0] != '.']
    for filename in filenames:
        src_path = os.path.join(input_dir, d, filename)
        dst_path = os.path.join(output_dir, 'svs', 'img_{:06d}.svs'.format(ind))
        os.system('ln -sf "{}" "{}"'.format(src_path, dst_path))
        svs_filenames.append(src_path)
        DX_filenames.append(dst_path)
        labels.append(ind)
df = pd.DataFrame({'orig_filename':svs_filenames, 'DX_filename': DX_filenames, 'group': dirs, 'response_groups': labels})
df.to_excel(os.path.join(output_dir, 'all.xlsx'))











