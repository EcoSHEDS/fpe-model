FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.2.0-gpu-py310-cu121-ubuntu20.04-ec2

# Install MLflow
RUN pip install --no-cache-dir mlflow

# Optional: Set environment variables for MLflow
# ENV MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
# ENV MLFLOW_EXPERIMENT_NAME=default

# Keep the original entrypoint
# ENTRYPOINT ["python", "-m", "torch_xla.distributed.run"] 