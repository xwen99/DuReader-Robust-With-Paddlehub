# coding:utf-8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning on reading comprehension task """

import argparse
import ast
import json
import os

import paddle
import paddle.fluid as fluid
import paddlehub as hub
from demo_dataset import DuReader
from my_reading_comprehension_task import ReadingComprehensionTask

hub.common.logger.logger.setLevel("INFO")

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--dataset_path", type=str, default=None, help="The diretory to DuReader robust dataset")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint.")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
args = parser.parse_args()
# yapf: enable.


if __name__ == '__main__':
    # 加载PaddleHub ERNIE预训练模型
    #module = hub.Module(name="ernie")
    module = hub.Module(name="chinese-roberta-wwm-ext-large")
    
    # ERNIE预训练模型输入变量inputs、输出变量outputs、以及模型program
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)

    # 加载竞赛数据集并使用ReadingComprehensionReader读取数据
    dataset = DuReader(dataset_path=args.dataset_path)
    reader = hub.reader.ReadingComprehensionReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_len=args.max_seq_len,
        doc_stride=128,
        max_query_length=64)

    # 取ERNIE的字级别预训练输出
    seq_output = outputs["sequence_output"]

    # 设置运行program所需的feed_list
    feed_list = [
        inputs["input_ids"].name,
        inputs["position_ids"].name,
        inputs["segment_ids"].name,
        inputs["input_mask"].name,
    ]

    # 设置运行配置
    config = hub.RunConfig(
        use_pyreader=True,
        use_data_parallel=False,
        use_cuda=args.use_gpu,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        strategy=hub.AdamWeightDecayStrategy())

    # 定义阅读理解Fine-tune Task
    # 由于竞赛数据集与cmrc2018数据集格式比较相似，此处sub_task应为cmrc2018
    # 否则运行可能出错
    reading_comprehension_task = ReadingComprehensionTask(
        data_reader=reader,
        feature=seq_output,
        feed_list=feed_list,
        config=config,
        sub_task="cmrc2018",
        max_answer_length=40
    )
    
    # 数据集测试集全部数据用于预测
    data = dataset.get_dev_examples()
    # 调用predict接口, 打开return_result(True)，将自动返回预测结果
    all_prediction, all_nbest_json = reading_comprehension_task.predict(data=data, load_best_model=True, return_result=True)
    # 写入预测结果
    json.dump(all_prediction, open(args.checkpoint_dir.split('/')[-1][5:] + '_prediction_dev.json', 'w'), ensure_ascii=False)
    json.dump(all_nbest_json, open(args.checkpoint_dir.split('/')[-1][5:] + '_nbest_dev.json', 'w'), ensure_ascii=False)