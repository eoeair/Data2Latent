import os
import jax.numpy as jnp
# The Flax NNX API.
from flax import nnx  
# optimizers
import optax
# checkpointer
import orbax.checkpoint as ocp

from net import CNN
from feeder import loader

def loss_fn(model: CNN, batch):
  latent, logits = model(batch['data'])
  loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch['label']).mean()
  return loss, logits

@nnx.jit
def train_step(model: CNN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.
  optimizer.update(model, grads)  # In-place updates.

@nnx.jit
def eval_step(model: CNN, metrics: nnx.MultiMetric, batch):
  loss, logits = loss_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.

@nnx.jit
def pred_step(model: CNN, batch):
  latent, logits = model(batch['data'])
  return logits.argmax(axis=1)

if __name__ == '__main__':
  # hyper param
  best_acc = 0
  batch_size = 256
  
  # Instantiate
  model = CNN(rngs=nnx.Rngs(0), dtype=jnp.bfloat16)
  tx = optax.adamw(learning_rate=0.01,b1=0.9)
  optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

  metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average('loss'),
  )
  
  train_loader = loader(dataset_path='data/mnist.npz', data='x_train', label='y_train', batch_size=batch_size, num_epochs=20)
  with ocp.CheckpointManager(
    os.path.join(os.getcwd(), 'checkpoints/'),
    options = ocp.CheckpointManagerOptions(max_to_keep=1),
    ) as mngr:
    for step, batch in enumerate(train_loader):
        train_step(model, optimizer, metrics, batch)
        if step > 0 and step % 1000 == 0:
            train_metrics = metrics.compute()
            print("Step:{}_Train Acc@1: {} loss: {} ".format(step,train_metrics['accuracy'],train_metrics['loss']))
            metrics.reset()  # Reset the metrics for the train set.

            # Compute the metrics on the test set after each training epoch.
            val_loader = loader(dataset_path='data/mnist.npz', data='x_val', label='y_val', batch_size=batch_size)
            for val_batch in val_loader:
                eval_step(model, metrics, val_batch)
            val_metrics = metrics.compute()
            print("Step:{}_Val Acc@1: {} loss: {} ".format(step,val_metrics['accuracy'],val_metrics['loss']))
            
            # save checkpoint
            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                _, state = nnx.split(model)
                mngr.save(step, args=ocp.args.StandardSave(state))
            metrics.reset()  # Reset the metrics for the val set.