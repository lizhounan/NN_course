from PIL import Image
import glob
filenames = glob.glob('*.jpg')
print(filenames)
# img = Image.open('image.png').convert('LA')
for file in filenames:
	img = Image.open(file).convert('LA')
	img.save('gray'+file[0] + '.png')