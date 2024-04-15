from pathlib import Path
import os
import shutil
os.mkdir("cc3m_5k")
path_list = Path("/data0/cc3m").rglob("*.jpeg")
i = 0
for path in path_list:
    if i > 10000:
        continue
    print(i, str(path), path.name)
    shutil.copy(str(path), os.path.join("cc3m_5k", path.name))
    i = i + 1
