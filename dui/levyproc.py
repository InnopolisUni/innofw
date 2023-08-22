import time
from tqdm import tqdm

for i in tqdm(range(0, 30)):
    if i%2==0:
        print(i)
    time.sleep(1)
