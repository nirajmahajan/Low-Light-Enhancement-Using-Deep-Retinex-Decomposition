import os
import matplotlib.pyplot as plt


for (root,dirs,files) in os.walk('.', topdown=True):
    for name in files:
    	if not name.endswith('.png'):
    		continue
    	path = os.path.join(root,name)
    	inp = plt.imread(path)
    	plt.imsave(path, inp[25:-40,170:-135,:])