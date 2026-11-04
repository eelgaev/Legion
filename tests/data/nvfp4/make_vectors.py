# Reference bytes from llama.cpp's converter for Legion's NVFP4 repack unit tests.
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np, torch, gguf
from conversion.base import ModelBase
g = torch.Generator().manual_seed(203)
rows, k = 3, 128
w = torch.randint(0, 256, (rows, k // 2), dtype=torch.uint8, generator=g)
s = torch.randint(0, 256, (rows, k // 16), dtype=torch.uint8, generator=g)
s[s == 0x7F] = 0x3F; s[s == 0xFF] = 0xBF  # keep the e4m3 bits finite
raw, shape = ModelBase._nvfp4_pack(w, s.view(torch.float8_e4m3fn))
print("shape", shape)
print("W", w.numpy().tobytes().hex())
print("S", s.numpy().tobytes().hex())
print("OUT", np.ascontiguousarray(raw).tobytes().hex())
# FP8 dequant to BF16 exactly as dequant_simple + the bf16 writer.
f8 = torch.arange(0, 256, dtype=torch.int32).to(torch.uint8)
f8[0x7F] = 0; f8[0xFF] = 0
scale = torch.tensor(0.0123456789, dtype=torch.float32)
deq = f8.view(torch.float8_e4m3fn).float() * scale
bf = gguf.quants.quantize(deq.numpy(), gguf.GGMLQuantizationType.BF16)
print("SCALE", scale.numpy().tobytes().hex())
print("BF16", bf.tobytes().hex())
