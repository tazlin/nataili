from .blip import Caption
from .clip import Interrogator
from .model_manager import (
    ModelManager,
    AITemplateModelManager,
    BlipModelManager,
    ClipModelManager,
    CodeFormerModelManager,
    CompVisModelManager,
    EsrganModelManager,
    GfpganModelManager,
)
from .util import logger
from .gfpgan import gfpgan
from .esrgan import esrgan
from .codeformers import codeformers
from .util import Switch
from .stable_diffusion import CompVis

disable_xformers = Switch()
disable_voodoo = Switch()
disable_local_ray_temp = Switch()
