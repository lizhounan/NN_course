from PIL import Image
import glob
import numpy as np
filenames = glob.glob('Gray/*.png')
print(filenames)

for file in filenames:
	arr = np.array(Image.open(file))[:, :, 0]
	for i in range(16):
		for j in range(16):
			arr[i][j] = 255 if arr[i][j] > 128 else 0

	img = Image.fromarray(arr)
	# img.show()
	img.save(file)