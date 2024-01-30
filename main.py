from PIL import Image
import torch
import numpy as np

torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)

repo = "isl-org/ZoeDepth"
# Zoe_N
model_zoe_n = torch.hub.load(repo, "ZoeD_NK", pretrained=True, config_mode="infer")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)


image = Image.open("./garbage3.jpg").convert("RGB")  # load
depth_numpy = zoe.infer_pil(image)  # as numpy

depth_pil = zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image


depth_tensor = zoe.infer_pil(image, output_type="tensor")  # as torch depth_tensor


depth_pil.save("output.png")
np.save("output.npy", depth_numpy)


print(depth_tensor.shape)
print(depth_tensor)
