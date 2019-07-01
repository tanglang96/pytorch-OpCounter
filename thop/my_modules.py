import argparse

import torch
import torch.nn as nn

class Conv2dSamePadding(nn.Conv2d):
	def __init__(self):
		super(self,Conv2dSamePadding).__init__()
