Run a distributed training/fine tuning job utilizing our Soperator solution nebius-solution-library/soperator at main · nebius/nebius-solution-library

Run inference on the same k8s cluster, that serving the aforementioned trained model

Run the aforementioned original (untrained) model, and compare the results of both models.

Utilize more than 80% of the GPUs (Nebius console includes monitoring dashboards)


Capacity limits: 

Training and Inference GPU limit = 2xH100s max

Single MK8s cluster at a time

Use only fabric-3 for the gpu_cluster resource.


Guidelines:

Avoid using the same shared filesystem for 2 different jails.

There’s an open issue in the Terraform recipe – in .tfvars file, turn this variable to false:

public_o11y_enabled = false

Install `yq` lib from the shell you’re running terraform apply

The dataset and model of choice can be anything.