# Transfer_learning_with_keras Training Module 

Tensorflow 2 supported
-----------------------------------------------------

How to save model and visualize on tensorboard.
===============================================
`Using Callbacks in model.fit()`
--------------------------------
```
callbacks = [
    ModelCheckpoint(
        os.path.join(checkpoint_path, CHECK_POINT),
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="auto"
        ),
    TensorBoard(log_dir= os.path.join(log_path,run_id) ),
    #lr_scheduler,
    ]
 ```
  * `ModelCheckpoint` - saving a model checkpoint `best model only` at every checkpoint.
  * `TensorBoard` -  Saving Training meta-data so we can visualize train and validation graph for `loss` and `accuracy`.
  
 command to visualize tensorboard.
  ---------------------------------
    $ tensorboard dev upload --logdir path-to-log

`Saving and loading model`
--------------------------
  * save model weights and model architecture in separate files.
  * save model architecture in both YAML and JSON format.
  * save model weights and architecture into a single file for later use.

