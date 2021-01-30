from PIL import Image
import glob
filenames = glob.glob('Gray/*.png')
print(filenames)

for file in filenames:
	img = Image.open(file).resize((16, 16))
	img.save(file)