from contextlib import ExitStack
from nerfbaselines import backends
from nerfbaselines.datasets import load_dataset
from nerfbaselines import (
    build_method_class,
)
from nerfbaselines.training import (
    Trainer, 
    Indices, 
    build_logger,
)
from nerfbaselines import (
    build_method_class,
    get_method_spec,
)

# We use the exit stack to simplify the context management
stack = ExitStack().__enter__()

# Prepare the output directory, and mount it if necessary (e.g., for docker backend)
output_path = "train_output"
stack.enter_context(backends.mount(output_path, output_path))

# Build the method class and start the backend
method_cls = stack.enter_context(build_method_class(method_spec, backend))

data = "external://blender/lego"

# Load train dataset
# We use the method info to load the required features and supported camera models
method_info = method_cls.get_method_info()
train_dataset = load_dataset(data, 
                             split="train", 
                             features=method_info.get("required_features"),
                             supported_camera_models=method_info.get("supported_camera_models"),
                             load_features=True)
# Load eval dataset
test_dataset = load_dataset(data, 
                            split="test", 
                            features=method_info.get("required_features"),
                            supported_camera_models=method_info.get("supported_camera_models"),
                            load_features=True)

# Build the method
model = method_cls(
    train_dataset=train_dataset,
    config_overrides=config_overrides,
)

# Build the trainer
trainer = Trainer(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    method=model,
    output=output_path,
    save_iters=Indices.every_iters(10_000, zero=True),  # Save the model every 10k iterations
    eval_few_iters=Indices.every_iters(2_000),  # Eval on few images every 2k iterations
    eval_all_iters=Indices([-1]),  # Eval on all images at the end
    logger=build_logger(("tensorboard",)),  # We will log to tensorboard
    generate_output_artifact=True,
    config_overrides=config_overrides,
    applied_presets=frozenset(presets),
)

# In this example we override the total number of iterations
# to make the training faster
trainer.num_iterations = 100

# Finally, we train the method
trainer.train()

# Close the stack. In real code, you should use the context manager
stack.close()