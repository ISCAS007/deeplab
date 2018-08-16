# -*- coding: utf-8 -*-

import tensorflow as tf
from deployment import model_deploy
from easydict import EasyDict as edict

# Set up deployment (i.e., multi-GPUs and/or multi-replicas).
FLAGS=edict()
FLAGS.num_clones=1
FLAGS.clone_on_cpu=False
FLAGS.task=0
FLAGS.num_replicas=1
FLAGS.num_ps_tasks=0

config = model_deploy.DeploymentConfig(
    num_clones=FLAGS.num_clones,
    clone_on_cpu=FLAGS.clone_on_cpu,
    replica_id=FLAGS.task,
    num_replicas=FLAGS.num_replicas,
    num_ps_tasks=FLAGS.num_ps_tasks)

print(config.inputs_device())
print(config.variables_device())
print(config.optimizer_device())