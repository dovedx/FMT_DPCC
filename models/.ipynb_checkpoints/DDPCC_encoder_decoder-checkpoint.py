import torch
import torch.nn as nn
import MinkowskiEngine as ME
from models.model_utils import *
from models.prior_module import*
from models.flow_loss import*

# from pytorch3d.ops import knn_points
from models.pointconv_util import*
import sys
sys.path.append('../')
from PointPWC.models import PointConvSceneFlowPWC8192selfglobalPointConv as PointConvSceneFlow

from PointPWC.models import multiScaleChamferSmoothCurvature