# transfer_learning_with_keras

## how to save model and visualize on tensorboard.
### `Using Callbacks in model.fit()`
  * `ModelCheckpoint` - saving a model checkpoint `best model only` at every checkpoint.
  * `TensorBoard` -  Saving Training meta-data so we can visualize train and validation graph for `loss` and `accuracy`.
     - command to visualize tensorboard : `tensorboard dev upload --logdir path-to-log-dir`
     

### `Saving and loading model`
  * save model weights and model architecture in separate files.
  * save model architecture in both YAML and JSON format.
  * save model weights and architecture into a single file for later use.

