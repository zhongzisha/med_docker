import os

dirs = os.listdir('/app/web/data')
dirs = [d+'\n' for d in dirs if d[0] != '.' or d != 'projects.txt']
with open('/app/web/data/projects.txt', 'w') as fp:
    fp.writelines(dirs)




