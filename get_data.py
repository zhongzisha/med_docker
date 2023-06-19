import pdb
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
responses = []
for ind, d in enumerate(dirs):
    filenames = os.listdir(os.path.join(input_dir, d))
    filenames = [filename for filename in filenames if filename[0] != '.']
    for ind1, filename in enumerate(filenames):
        src_path = os.path.join(input_dir, d, filename)
        dst_path = os.path.join(output_dir, 'svs', 'img_{:06d}_{:06d}.svs'.format(ind, ind1))
        os.system('ln -sf "{}" "{}"'.format(src_path, dst_path))
        svs_filenames.append(src_path)
        DX_filenames.append(dst_path)
        labels.append(ind)
        responses.append(d)
print(len(svs_filenames))
print(len(DX_filenames))
print(len(responses))
print(len(labels))

df = pd.DataFrame({'orig_filename':svs_filenames, 'DX_filename': DX_filenames, 'response': responses, 'response_groups': labels})
df.to_excel(os.path.join(output_dir, 'all.xlsx'))











