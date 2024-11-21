from openood.utils import Config

from .ash_postprocessor import ASHPostprocessor
from .base_postprocessor import BasePostprocessor
from .cider_postprocessor import CIDERPostprocessor
from .conf_branch_postprocessor import ConfBranchPostprocessor
from .cutpaste_postprocessor import CutPastePostprocessor
from .dice_postprocessor import DICEPostprocessor
from .draem_postprocessor import DRAEMPostprocessor
from .dropout_postprocessor import DropoutPostProcessor
from .dsvdd_postprocessor import DSVDDPostprocessor
from .ebo_postprocessor import EBOPostprocessor
from .ensemble_postprocessor import EnsemblePostprocessor
from .gmm_postprocessor import GMMPostprocessor
from .godin_postprocessor import GodinPostprocessor
from .gradnorm_postprocessor import GradNormPostprocessor
from .gram_postprocessor import GRAMPostprocessor
from .kl_matching_postprocessor import KLMatchingPostprocessor
from .knn_postprocessor import KNNPostprocessor
from .maxlogit_postprocessor import MaxLogitPostprocessor
from .mcd_postprocessor import MCDPostprocessor
from .mds_postprocessor import MDSPostprocessor
from .mds_ensemble_postprocessor import MDSEnsemblePostprocessor
from .mos_postprocessor import MOSPostprocessor
from .npos_postprocessor import NPOSPostprocessor
from .odin_postprocessor import ODINPostprocessor
from .opengan_postprocessor import OpenGanPostprocessor
from .openmax_postprocessor import OpenMax
from .patchcore_postprocessor import PatchcorePostprocessor
from .rd4ad_postprocessor import Rd4adPostprocessor
from .react_postprocessor import ReactPostprocessor
from .rmds_postprocessor import RMDSPostprocessor
from .residual_postprocessor import ResidualPostprocessor
from .rotpred_postprocessor import RotPredPostprocessor
from .rankfeat_postprocessor import RankFeatPostprocessor
from .ssd_postprocessor import SSDPostprocessor
from .she_postprocessor import SHEPostprocessor
from .temp_scaling_postprocessor import TemperatureScalingPostprocessor
from .vim_postprocessor import VIMPostprocessor
from .rts_postprocessor import RTSPostprocessor
from .gen_postprocessor import GENPostprocessor
from .relation_postprocessor import RelationPostprocessor
from .mcm_postprocessor import MCMPostprocessor
from .ttaprompt_postprocessor import TTAPromptPostprocessor, TTAPromptLocalfeatPostprocessor, TTAPromptPostprocessor_noadagap
from .oneoodprompt_postprocessor import OneOodPromptPostprocessor, OneOodPromptDevelopPostprocessor
from .she_oneoodprompt_postprocessor import SHEOneOodPromptPostprocessor
from .knn_oneoodprompt_postprocessor import KnnOneOodPromptPostprocessor
from .label_relationship_postprocessor import LabelRelationPostprocessor



def get_postprocessor(config: Config):
    postprocessors = {
        'ash': ASHPostprocessor,
        'cider': CIDERPostprocessor,
        'conf_branch': ConfBranchPostprocessor,
        'msp': BasePostprocessor,
        'ebo': EBOPostprocessor,
        'odin': ODINPostprocessor,
        'mds': MDSPostprocessor,
        'mds_ensemble': MDSEnsemblePostprocessor,
        'rmds': RMDSPostprocessor,
        'gmm': GMMPostprocessor,
        'patchcore': PatchcorePostprocessor,
        'openmax': OpenMax,
        'react': ReactPostprocessor,
        'vim': VIMPostprocessor,
        'gradnorm': GradNormPostprocessor,
        'godin': GodinPostprocessor,
        'gram': GRAMPostprocessor,
        'cutpaste': CutPastePostprocessor,
        'mls': MaxLogitPostprocessor,
        'npos': NPOSPostprocessor,
        'residual': ResidualPostprocessor,
        'klm': KLMatchingPostprocessor,
        'temperature_scaling': TemperatureScalingPostprocessor,
        'ensemble': EnsemblePostprocessor,
        'dropout': DropoutPostProcessor,
        'draem': DRAEMPostprocessor,
        'dsvdd': DSVDDPostprocessor,
        'mos': MOSPostprocessor,
        'mcd': MCDPostprocessor,
        'opengan': OpenGanPostprocessor,
        'knn': KNNPostprocessor,
        'dice': DICEPostprocessor,
        'ssd': SSDPostprocessor,
        'she': SHEPostprocessor,
        'rd4ad': Rd4adPostprocessor,
        'rts': RTSPostprocessor,
        'rotpred': RotPredPostprocessor,
        'rankfeat': RankFeatPostprocessor,
        'gen': GENPostprocessor,
        'relation': RelationPostprocessor,
        'mcm': MCMPostprocessor,
        'oneoodprompt':OneOodPromptPostprocessor,
        'oneoodpromptdevelop':OneOodPromptDevelopPostprocessor,
        'sheoneoodprompt': SHEOneOodPromptPostprocessor,
        'knnoneoodprompt': KnnOneOodPromptPostprocessor,
        'ttaprompt': TTAPromptPostprocessor,
        'ttapromptnoadagap': TTAPromptPostprocessor_noadagap,
        'ttapromptlocalfeat': TTAPromptLocalfeatPostprocessor,
        'labelrelationship': LabelRelationPostprocessor
    }

    return postprocessors[config.postprocessor.name](config)
