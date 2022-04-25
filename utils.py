from torchsummary import summary
from network import *

args = VARIANT
vae = Vae(args)
summary(vae, input_size=(3, 400, 600))