import soundcard as sc
import numpy as np
import resampy
import tensorflow as tf
import tensorflow_hub as hub

print("All imports OK")
print("Default speaker:", sc.default_speaker())
mics = sc.all_microphones(include_loopback=True)
print("\nLoopback devices available:")
for m in mics:
    print(" -", m.name)