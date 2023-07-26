import argparse
import ast
import logging
import os
import torch
# import json

JSON_CONTENT_TYPE = 'application/json'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def train(args):
    logger.debug("training starting")
    logger.debug(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device Type: {}".format(device))

    # model = BERTopic(language=args.language)

    # logger.info("BERTtopic Model loaded for language {}".format(args.language))

    print("loading: training data")
    # docs = []
    print(f"data_dir: {args.data_dir}")
    # with open(args.data_dir+"/training_file.txt") as file:
    #     for line in file:
    #         docs.append(line.rstrip())

    print("fitting model")
    # topics, probs = model.fit_transform(docs)
    print("finished")
    # return _save_model(model, args.model_dir)


def _save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "my_model")
    model.save(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--custom", type=str, default="streamflow", help="dummy custom argument"
    )

    # The parameters below retrieve their default values from SageMaker environment variables, which are
    # instantiated by the SageMaker containers framework.
    # https://github.com/aws/sagemaker-containers#how-a-script-is-executed-inside-the-container
    parser.add_argument("--hosts", type=str, default=ast.literal_eval(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    train(parser.parse_args())