from __future__ import annotations
import argparse, json, os, sys, time, struct, numpy as np
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt

from tinygrad import Tensor, nn
from tinygrad.nn.optim import Adam
from tinygrad.nn import Conv1d, BatchNorm1d, Dropout
from deep_signal_cnn import DeepSignalCNN

class BinaryDataset:
    def __init__(self, path: str, k: int, s: int):
        self.k, self.s = k, s
        self.record_bytes = k + 4*k + 4*k + 2*k + 4*s + 1
        self._fd = open(path, "rb")
        self._fd.seek(0, 2)
        self._n_records = self._fd.tell() // self.record_bytes
        self._fd.seek(0)

    def __len__(self): return self._n_records

    def read_one(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray, int]:
        self._fd.seek(idx * self.record_bytes)
        raw = self._fd.read(self.record_bytes)
        unpack = struct.unpack
        k, s = self.k, self.s
        bases     = np.frombuffer(raw[0:k], dtype=np.uint8)
        means     = np.frombuffer(raw[k:k+4*k], dtype=np.float32)
        stds      = np.frombuffer(raw[k+4*k:k+8*k], dtype=np.float32)
        sanums    = np.frombuffer(raw[k+8*k:k+10*k], dtype=np.uint16)
        signals   = np.frombuffer(raw[k+10*k:k+10*k+4*s], dtype=np.float32)
        label     = raw[-1]
        return bases, means, stds, sanums, signals, int(label)

    def batch(self, indices: np.ndarray) -> Tuple[Tensor, Tensor>:
        
        signals_lst, labels_lst = [], []
        for i in indices:
            _, _, _, _, sig, lab = self.read_one(i)
            signals_lst.append(sig)
            labels_lst.append(lab)
        
        signals_np = np.stack(signals_lst, axis=0)[:, None, :]
        labels_np  = np.array(labels_lst, dtype=np.float32)
        return Tensor(signals_np), Tensor(labels_np)

    def shuffle_batch_iterator(self, batch_size: int):
        
        indices = np.arange(len(self))
        while True:
            np.random.shuffle(indices)
            for start in range(0, len(self), batch_size):
                end = min(start + batch_size, len(self))
                yield self.batch(indices[start:end])
            


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))

def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return float(tp / (tp + fn + 1e-8))

def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(tp / (tp + fp + 1e-8))


def train_one_epoch(model, dataset, batch_size, opt, display_step: int,
                    epoch_id: int, log_dir: Path):
    model.train()
    itr = dataset.shuffle_batch_iterator(batch_size)
    epoch_losses, streaming_losses = [], []

    for iter_id, (x, y) in enumerate(itr, 1):
        
        logits = model(x)                       
        logits = logits.mean(axis=1, keepdim=True)  
        loss = logits.binary_crossentropy_logits(y).mean()

        
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_item = loss.item()
        epoch_losses.append(loss_item)
        streaming_losses.append(loss_item)

        if iter_id % display_step == 0:
            avg_streaming_loss = np.mean(streaming_losses)
            print(f"epoch {epoch_id:03d} iter {iter_id:05d} | loss {avg_streaming_loss:.4f}")

            
            val_loss = evaluate(model, valid_ds, batch_size)
            print(f"          valid | loss {val_loss:.4f}")

            
            if log_dir:
                (log_dir / "train.log").open("a").write(
                    json.dumps({"epoch": epoch_id, "iter": iter_id, "loss": avg_streaming_loss}) + "\n")
                (log_dir / "valid.log").open("a").write(
                    json.dumps({"epoch": epoch_id, "iter": iter_id, "loss": val_loss}) + "\n")

            
            streaming_losses = []

    return np.mean(epoch_losses) if epoch_losses else 0.


@Tensor.no_grad()
def evaluate(model, dataset, batch_size):
    model.eval()
    val_iter = dataset.shuffle_batch_iterator(batch_size)  
    val_loss = []
    n = 0
    for x, y in val_iter:
        logits = model(x)
        logits = logits.mean(axis=1, keepdim=True)
        loss = logits.binary_crossentropy_logits(y).mean()

        val_loss.append(loss.item())
        n += y.shape[0]
        if n >= len(dataset):
            break
    return float(np.mean(val_loss))


def parse_args():
    p = argparse.ArgumentParser("Train DeepSignal with TinyGrad")
    # input
    p.add_argument("--train_file", required=True, help="binary file (train)")
    p.add_argument("--valid_file", required=True, help="binary file (valid)")
    p.add_argument("--kmer_len", type=int, default=17)
    p.add_argument("--cent_signals_len", type=int, default=360)
    # output
    p.add_argument("--model_dir", required=True)
    p.add_argument("--log_dir", default=None)
    # train
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--learning_rate", type=float, default=0.001)
    p.add_argument("--decay_rate", type=float, default=0.1)
    p.add_argument("--max_epoch_num", type=int, default=10)
    p.add_argument("--min_epoch_num", type=int, default=5)
    p.add_argument("--display_step", type=int, default=100)
    p.add_argument("--keep_prob", type=float, default=0.5)
    # booleans (kept for CLI compatibility)
    p.add_argument("--is_binary", default="yes", choices=["yes", "no"])
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    global train_ds, valid_ds
    train_ds = BinaryDataset(args.train_file, args.kmer_len, args.cent_signals_len)
    valid_ds = BinaryDataset(args.valid_file, args.kmer_len, args.cent_signals_len)

    model = DeepSignalCNN(dropout_p=1.0 - args.keep_prob)
    opt = Adam(nn.state.get_parameters(model), lr=args.learning_rate)

    train_losses_per_epoch = []
    valid_losses_per_epoch = []
    best_val_loss = float('inf')

    for epoch in range(1, args.max_epoch_num + 1):
        lr = args.learning_rate * (args.decay_rate ** max(0, epoch - 2))
        opt.lr = lr
        
        avg_train_loss = train_one_epoch(model, train_ds, args.batch_size, opt,
                                         args.display_step, epoch, Path(args.log_dir) if args.log_dir else None)
        train_losses_per_epoch.append(avg_train_loss)

        val_loss = evaluate(model, valid_ds, args.batch_size)
        valid_losses_per_epoch.append(val_loss)
        
        print(f"Epoch {epoch:03d} summary | valid loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            nn.state.safe_save(nn.state.get_state_dict(model),
                               os.path.join(args.model_dir, f"best_epoch_{epoch}.pt"))
            print(f"  -> new best saved (loss={val_loss:.4f})")
        
        if epoch >= args.min_epoch_num and val_loss > best_val_loss + 0.01:
            print("Early stopping")
            break

    print("Training finished. Best validation loss = {:.4f}".format(best_val_loss))

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses_per_epoch, label='Training Loss')
    plt.plot(valid_losses_per_epoch, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    print("Loss plot saved to loss_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
