# Copyright 2019 Google Inc. All Rights Reserved.
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
# ==============================================================================

#!/usr/bin/env bash
python -u run.py --is_training True --dataset_name movie \
    --train_data_paths data/movie --valid_data_paths data/movie --save_dir checkpoints/movie \
    --gen_frm_dir results/movie --model_name e3d_lstm --allow_gpu_growth True\
    --img_channel 3 --img_width 256 --input_length 4 --total_length 5 --filter_size 5 \
    --num_hidden 64,64,64,64 --patch_size 8 --layer_norm True --reverse_input False --sampling_stop_iter 100000 \
    --sampling_start_value 1.0 --sampling_delta_per_iter 0.00001 --lr 0.001 --batch_size 2 --max_iterations 100 \
    --display_interval 1 --test_interval 1 --snapshot_interval 5000
    --num_save_samples 1000

python -u run.py --is_training True --dataset_name movie --train_data_paths data/movie --valid_data_paths data/movie --save_dir checkpoints/movie --gen_frm_dir results/movie --model_name e3d_lstm --allow_gpu_growth True--img_channel 3 --img_width 256 --input_length 4 --total_length 5 --filter_size 5 --num_hidden 64,64,64,64 --patch_size 8 --layer_norm True --reverse_input False --sampling_stop_iter 100000 --sampling_start_value 1.0 --sampling_delta_per_iter 0.00001 --lr 0.001 --batch_size 2 --max_iterations 100 --display_interval 1 --test_interval 1 --snapshot_interval 5000