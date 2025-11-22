import logging
import traceback

from datasets import load_dataset

from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.evaluation import CrossEncoderCorrelationEvaluator
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

DATASET_PATH = "dataset.jsonl"

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

model_name = "cross-encoder/stsb-roberta-base"
train_batch_size = 1
num_epochs = 1

# 1a. Load a model to finetune with 1b. (Optional) model card data
model = CrossEncoder(model_name)
print("Model max length:", model.max_length)
print("Model num labels:", model.num_labels)

logging.info("Read the gooaq training dataset")
dataset = load_dataset("json", data_files = DATASET_PATH, split="train")

dataset_dict = dataset.train_test_split(test_size=2, seed=12)
train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["test"]
eval_dataset = eval_dataset.rename_columns({
    "query": "sentence1",
    "response": "sentence2",
    "label": "score"
})
logging.info(train_dataset)
logging.info(eval_dataset)

# 3. Define our training loss.
loss = BinaryCrossEntropyLoss(model=model)

pairs = list(zip(eval_dataset["sentence1"], eval_dataset["sentence2"]))

evaluator = CrossEncoderCorrelationEvaluator(
    sentence_pairs=pairs,
    scores=eval_dataset["score"],
    name="sts_dev",
)
evaluator(model)

# 5. Define the training arguments
run_name = f"reranker-"
args = CrossEncoderTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    # Optional training parameters:
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=50,
    logging_first_step=True,
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
    seed=12,
)

# 6. Create the trainer & start training
trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=evaluator,
)
trainer.train()

# 7. Evaluate the final model, useful to include these in the model card
evaluator(model)

# 8. Save the final model
final_output_dir = f"models/{run_name}/final"
model.save_pretrained(final_output_dir)

