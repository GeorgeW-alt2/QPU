import random
rows = 100000
with open("data.csv", mode='w') as f:
    f.write('\n'.join(str(i)for i in range(rows)))
     
