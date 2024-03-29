# %reload_ext autoreload
# %autoreload 2
import imageio
import matplotlib.pyplot as plt
import torch
print(torch.cuda.is_available())
import sys
sys.path.append("D:/SATR/GLIP")

from meshseg.models.GLIP.glip import GLIPModel
GM = GLIPModel()


si = 128
# img = torch.zeros((si, si, 3), dtype=torch.uint8).numpy()
img = imageio.imread("D:/SATR/data/imgOIP.jpg")
# img += 2
# plt.imshow(img)
print(img.shape)
# prediction = GM.predict(img, "The legs, head, eyes, arms of people")
# prediction = GM.predict(img, "the legs, top of the table")
prediction = GM.predict(img, "legs")
# prediction = GM.predict(img, "")


print(prediction[1])
plt.figure(figsize=[15, 10])
plt.imshow(prediction[0])
plt.show()