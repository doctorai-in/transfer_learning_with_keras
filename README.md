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
    TensorBoard(log_dir= os.path.join(log_path,run_id) )
    ]
 ```
  * `ModelCheckpoint` - saving a model checkpoint `best model only` at every checkpoint.
  * `TensorBoard` -  Saving Training meta-data so we can visualize train and validation graph for `loss` and `accuracy`.
  
 command to visualize tensorboard.
  ---------------------------------
    $ tensorboard dev upload --logdir path-to-log

`Saving and loading model`
--------------------------
- [Refer to article](https://machinelearningmastery.com/save-load-keras-deep-learning-models/)
* save model weights and model architecture in separate files.
* save model architecture in both YAML and JSON format.
* save model weights and architecture into a single file for later use.

`Saving History loss and accuracy in csv`
-----------------------------------------
```
hist = model.fit(
    train_generator,
    callbacks = callbacks,
    batch_size=16,
    steps_per_epoch = 10,
    epochs=nb_epochs,
    validation_data=eval_generator,
    shuffle=True,
    validation_steps=2,
    )
pd.DataFrame(hist.history).to_csv(os.path.join(destination,HISTORY_FILE))
```


