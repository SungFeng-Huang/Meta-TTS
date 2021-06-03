from PIL import Image

im1 = Image.open('images/LibriTTS/det.png')
im2 = Image.open('images/VCTK/det.png')
w, h = im1.size
pad = 0
crop_w = int(w*0.68)
im = Image.new('RGB', (crop_w+pad+w, h), (255,255,255))
im.paste(im1, (0, 0))
im.paste(im2.crop((0, 0, crop_w, h)), (w+pad,0))
im.save('images/det.png')
im.show()

# im1 = Image.open('images/LibriTTS/eer.png')
# im2 = Image.open('images/VCTK/eer.png')
# w, h = im1.size
# pad = 0
# crop_w = int(w*0.66)
# im = Image.new('RGB', (crop_w+pad+w, h), (255,255,255))
# im.paste(im1, (0,0))
# im.paste(im2.crop((0, 0, crop_w, h)), (w+pad, 0))
# im.save('images/eer.png')
# im.show()

# im1 = Image.open('images/LibriTTS/errorbar_plot.png')
# im2 = Image.open('images/VCTK/errorbar_plot.png')
# w, h = im1.size
# pad = 0
# crop_w = int(w*0.66)
# im = Image.new('RGB', (crop_w+pad+w, h), (255,255,255))
# im.paste(im1, (0,0))
# im.paste(im2.crop((0, 0, crop_w, h)), (w+pad, 0))
# im.save('images/errorbar_plot.png')
# im.show()

# im1 = Image.open('images/LibriTTS/roc.png')
# im2 = Image.open('images/VCTK/roc.png')
# w, h = im1.size
# pad = 0
# crop_w = int(w*0.68)
# im = Image.new('RGB', (crop_w+pad+w, h), (255,255,255))
# im.paste(im1, (0, 0))
# im.paste(im2.crop((0, 0, crop_w, h)), (w+pad,0))
# im.save('images/roc.png')
# im.show()
