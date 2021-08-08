# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import paddle
from paddle import nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
# from mayavi_visual import *


@manager.LOSSES.add_component
class MixedLoss(nn.Layer):
    """
    Weighted computations for multiple Loss.
    The advantage is that mixed loss training can be achieved without changing the networking code.

    Args:
        losses (list[nn.Layer]): A list consisting of multiple loss classes
        coef (list[float|int]): Weighting coefficient of multiple loss

    Returns:
        A callable object of MixedLoss.
    """

    def __init__(self, losses, coef):
        super(MixedLoss, self).__init__()
        if not isinstance(losses, list):
            raise TypeError('`losses` must be a list!')
        if not isinstance(coef, list):
            raise TypeError('`coef` must be a list!')
        len_losses = len(losses)
        len_coef = len(coef)
        if len_losses != len_coef:
            raise ValueError(
                'The length of `losses` should equal to `coef`, but they are {} and {}.'
                .format(len_losses, len_coef))

        self.losses = losses
        self.coef = coef

    def forward(self, logits, labels):
        # bs,h,w = labels.shape
        # p_r = (labels>0).astype('float32').reshape([bs,h*w]).sum(-1) / paddle.ones_like(labels[0]).sum()
        # p_r_dict = {}
        # p_r_dict[0] = 3
        # p_r_dict[1] = 4
        # p_r_dict[2] = 5
        # p_r_dict[3] = 6
        # p_r_dict[4] = -1
        # p_r_dict[5] = -1
        # p_r_dict[6] = -1
        # p_r_dict[7] = -1
        # p_r_dict[8] = -1
        # p_r_dict[9] = -1
        # p_r_dict[10] = -1

        # # print('p_r=', p_r)

        # ignore_mask = (labels>0).clone().unsqueeze(1).astype('float32')
        # ignore_mask = nn.AdaptiveMaxPool2D((32, 32))(ignore_mask)


        # # 膨胀
        # delatetion = 3
        # constant_kernel = paddle.ones([1, 1, delatetion, delatetion])
        # constant_kernel.stop_gradient = True
        # delatetion_count = [1, 2, 3, 3, 4]

        # ignore_mask_list = []
        # for i in range(14):
        #     ignore_mask = F.conv2d(ignore_mask, constant_kernel, padding=int(delatetion / 2))
        #     ignore_mask_list.append(ignore_mask)

        # for i in range (bs):
        #     # print('int(p_r[i]*10)=', int(p_r[i]*10))
        #     ignore_mask[i] = ignore_mask_list[p_r_dict[int(p_r[i]*10)]][i]

        # ignore_mask = (ignore_mask > 0.0).astype('float32')
        # ignore_mask = nn.UpsamplingNearest2D(labels.shape[-2:])(ignore_mask)
        # ignore_mask = ignore_mask.squeeze(1).detach()

        # # print('np.random.uniform(0,1)=', np.random.randint(0, 100) / 100)
        # # orin_rgb = (labels[0]>0).astype('float32').cpu().numpy()
        # # orin_label = ignore_mask[0].astype('float32').cpu().numpy()
        # # # plt.figure()
        # # # plt.imshow(orin_rgb)
        # # # plt.show()
        # # plt_show([orin_rgb, orin_label])

        # # input('kkk')

        # negative_one = paddle.ones_like(ignore_mask)*255
        # ignore_label_mask = paddle.where(ignore_mask > 0, labels.astype('float32'), negative_one)


        # if np.random.randint(0, 100) / 100 > 0.5:
            # ignore_label_mask = labels.detach()
        ignore_label_mask = labels.detach()
        loss_list = []
        final_output = 0
        for i, loss in enumerate(self.losses):
            output = loss(logits, ignore_label_mask.astype('int64'))
            final_output += output * self.coef[i]

        return final_output
