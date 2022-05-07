# Copyright 2017 Natural Language Processing Group, Nanjing University, zhengzx.142857@gmail.com.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from .transformer_8channel_ctc import Covn2dTransformerCTC
from .mitlstmctc import MITLSTM4ETCTC
from .vaetransformer_8channel import VaeCovn2dTransformer
from .vae_joint_transformer_move_node import VaeCovn2dTransformerMoveNode
from .transformer_move_node_8channel_ctc import MovenodeCovn2dTransformerCTC
# from .emglstm import EMGLSTM


__all__ = [
    "build_model",
]
# "WavenetLSTM": WavenetLSTM,
MODEL_CLS = {

    'Covn2dTransformerCTC':Covn2dTransformerCTC,
    'MITLSTM4ETCTC': MITLSTM4ETCTC,
    'VaeCovn2dTransformer': VaeCovn2dTransformer,
    'VaeCovn2dTransformerMoveNode': VaeCovn2dTransformerMoveNode,
    'MovenodeCovn2dTransformerCTC': MovenodeCovn2dTransformerCTC,
    }


def build_model(model: str, **kwargs):
    if model not in MODEL_CLS:
        raise ValueError(
            "Invalid model class \'{}\' provided. Only {} are supported now.".format(
                model, list(MODEL_CLS.keys())))

    return MODEL_CLS[model](**kwargs)