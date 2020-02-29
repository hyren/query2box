#!/bin/bash

 CUDA_VISIBLE_DEVICES=0 python3.5 -u codes/run.py --do_train --cuda --do_valid --do_test \
  --data_path data/FB15k --model BoxTransE -n 128 -b 512 -d 400 -g 24 -a 1.0 \
  -lr 0.0001 --max_steps 300000 --cpu_num 1 --test_batch_size 16 --center_reg 0.02 \
  --geo box --task 1c.2c.3c.2i.3i.ic.ci.2u.uc --stepsforpath 300000  --offset_deepsets inductive --center_deepsets eleattention \
  --print_on_screen

 CUDA_VISIBLE_DEVICES=1 python3.5 -u codes/run.py --do_train --cuda --do_valid --do_test \
  --data_path data/FB15k-237 --model BoxTransE -n 128 -b 512 -d 400 -g 24 -a 1.0 \
  -lr 0.0001 --max_steps 300000 --cpu_num 1 --test_batch_size 16 --center_reg 0.02 \
  --geo box --task 1c.2c.3c.2i.3i.ic.ci.2u.uc --stepsforpath 300000  --offset_deepsets inductive --center_deepsets eleattention \
  --print_on_screen

 CUDA_VISIBLE_DEVICES=2 python3.5 -u codes/run.py --do_train --cuda --do_valid --do_test \
  --data_path data/NELL --model BoxTransE -n 128 -b 512 -d 400 -g 24 -a 1.0 \
  -lr 0.0001 --max_steps 300000 --cpu_num 1 --test_batch_size 16 --center_reg 0.02 \
  --geo box --task 1c.2c.3c.2i.3i.ic.ci.2u.uc --stepsforpath 300000  --offset_deepsets inductive --center_deepsets eleattention \
  --print_on_screen
