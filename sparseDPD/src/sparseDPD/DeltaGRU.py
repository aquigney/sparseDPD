"""
DeltaGRU forward/inverse PA model wrapper.

This is a *standalone* class (you do NOT need to modify your existing NeuralNetwork class).
It exposes a similar API surface:
  - get_best_model(...)
  - generate_model_output(...)
  - calculate_forward_nmse(...)

Key differences vs your PNTDNN wrapper:
  - DeltaGRU trains on SEQUENCES: tensors shaped (batch, time, 2) where last dim is [I, Q]
  - It slices long captures into windows of length seq_len with hop seq_hop.

Expected Dataset interface (same as yours):
  - dataset.input_data : 1D complex numpy array
  - dataset.output_data: 1D complex numpy array
  - Dataset.conj_phase(sig) : returns complex phase reference (same length as sig)

Usage:
  model = DeltaGRUNetwork(hidden_size=64, num_layers=1, thx=0.0, thh=0.0,
                          forward_model=True, seq_len=1024, seq_hop=1024)
  train_losses, valid_losses, best_epoch = model.get_best_model(200, train_ds, valid_ds, learning_rate=1e-3)
  yhat = model.generate_model_output(test_ds.input_data)   # complex numpy array
"""

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .Dataset import Dataset


# ----------------------------
# Core DeltaGRU model (self-contained; no external deps)
# ----------------------------

class DeltaGRUModel(nn.Module):
    """
    input:  (B, T, 2)  where last dim is [I, Q]
    output: (B, T, 2)  predicted [I, Q] (normalized domain)
    """
    def __init__(self, hidden_size=64, num_layers=1, thx=0.0, thh=0.0, use_tcn_skip=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.thx = thx
        self.thh = thh

        # Feature vector inside DeltaGRU is 6: [i, q, amp, amp^3, i_prev, q_prev]
        self.rnn = DeltaGRULayer(
            input_size=6,
            hidden_size=hidden_size,
            num_layers=num_layers,
            thx=thx,
            thh=thh,
        )

        self.fc_out = nn.Linear(hidden_size, 2, bias=False)

        self.use_tcn_skip = use_tcn_skip
        if use_tcn_skip:
            self.tcn = nn.Sequential(
                nn.Conv1d(2, 3, kernel_size=3, padding=16, dilation=16, bias=False),
                nn.Hardswish(),
                nn.Conv1d(3, 2, kernel_size=1, padding=0, dilation=1, bias=False),
                nn.Hardswish(),
            )
        else:
            self.tcn = None

    def init_hidden(self, batch_size: int, device, dtype=torch.float32):
        # h_0 in DeltaGRULayer: (L, B, H)
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
        """
        x:  (B,T,2)
        h0: (L,B,H) or None
        """
        if x.ndim != 3 or x.size(-1) != 2:
            raise ValueError(f"DeltaGRUModel expects x shaped (B,T,2). Got {tuple(x.shape)}")

        B, T, _ = x.shape
        device = x.device

        if h0 is None:
            h0 = self.init_hidden(B, device=device, dtype=x.dtype)

        skip = 0.0
        if self.tcn is not None:
            skip = self.tcn(x.transpose(1, 2)).transpose(1, 2)  # (B,T,2)

        i = x[..., 0:1]
        q = x[..., 1:2]
        amp2 = i * i + q * q
        amp = torch.sqrt(torch.clamp(amp2, min=1e-12))
        amp3 = amp * amp2

        # previous-sample features (no wrap-around)
        i_prev = torch.zeros_like(i)
        q_prev = torch.zeros_like(q)
        i_prev[:, 1:, :] = i[:, :-1, :]
        q_prev[:, 1:, :] = q[:, :-1, :]

        feats = torch.cat([i, q, amp, amp3, i_prev, q_prev], dim=-1)  # (B,T,6)

        # IMPORTANT: call with keyword argument for h_0
        out = self.rnn(feats, h_0=h0)  # (B,T,H)
        y = self.fc_out(out) + skip    # (B,T,2)
        return y


class DeltaGRULayer(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=1, thx=0.0, thh=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.th_x = float(thx)
        self.th_h = float(thh)

        self.weight_ih_height = 3 * hidden_size
        self.x_p_length = max(input_size, hidden_size)
        self.batch_first = True

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def _process_inputs(self, x, x_p_0=None, h_0=None, h_p_0=None, dm_ch_0=None, dm_0=None):
        # x: (B,T,F) -> (T,B,F)
        if self.batch_first:
            x = x.transpose(0, 1)
        T, B, F = x.shape

        if x_p_0 is None or h_0 is None or h_p_0 is None or dm_ch_0 is None or dm_0 is None:
            x_p_0 = torch.zeros(self.num_layers, B, self.x_p_length, dtype=x.dtype, device=x.device)
            h_0   = torch.zeros(self.num_layers, B, self.hidden_size, dtype=x.dtype, device=x.device)
            h_p_0 = torch.zeros(self.num_layers, B, self.hidden_size, dtype=x.dtype, device=x.device)
            dm_ch_0 = torch.zeros(self.num_layers, B, self.hidden_size, dtype=x.dtype, device=x.device)
            dm_0 = torch.zeros(self.num_layers, B, self.weight_ih_height, dtype=x.dtype, device=x.device)

        return x, x_p_0, h_0, h_p_0, dm_ch_0, dm_0

    @staticmethod
    def _compute_deltas(x, x_p, h, h_p, th_x, th_h):
        dx = x - x_p
        dh = h - h_p

        dx_abs = dx.abs()
        dh_abs = dh.abs()

        dx = dx.masked_fill(dx_abs < th_x, 0.0)
        dh = dh.masked_fill(dh_abs < th_h, 0.0)
        return dx, dh, dx_abs, dh_abs

    @staticmethod
    def _update_states(dx_abs, dh_abs, x, h, x_p, h_p, th_x, th_h):
        x_p = torch.where(dx_abs >= th_x, x, x_p)
        h_p = torch.where(dh_abs >= th_h, h, h_p)
        return x_p, h_p

    def _compute_gates(self, dx, dh, dm, dm_nh):
        mac_x = self.x2h(dx) + dm
        mac_h = self.h2h(dh)

        mac_x_r, mac_x_z, mac_x_n = mac_x.chunk(3, dim=1)
        mac_h_r, mac_h_z, mac_h_n = mac_h.chunk(3, dim=1)

        dm_r = mac_x_r + mac_h_r
        dm_z = mac_x_z + mac_h_z
        dm_n = mac_x_n
        dm_nh = mac_h_n + dm_nh

        dm = torch.cat([dm_r, dm_z, dm_n], dim=1)
        return dm, dm_r, dm_z, dm_n, dm_nh

    def _layer_forward(self, x_TBF, x_p, h, h_p, dm_nh, dm):
        th_x = torch.tensor(self.th_x, dtype=x_TBF.dtype, device=x_TBF.device)
        th_h = torch.tensor(self.th_h, dtype=x_TBF.dtype, device=x_TBF.device)

        T, B, F = x_TBF.shape
        out = []

        for t in range(T):
            x = x_TBF[t]  # (B,F)

            dx, dh, dx_abs, dh_abs = self._compute_deltas(x, x_p, h, h_p, th_x, th_h)
            x_p, h_p = self._update_states(dx_abs, dh_abs, x, h, x_p, h_p, th_x, th_h)

            dm, dm_r, dm_z, dm_n, dm_nh = self._compute_gates(dx, dh, dm, dm_nh)

            r = self.sigmoid(dm_r)
            z = self.sigmoid(dm_z)
            n = self.tanh(dm_n + r * dm_nh)

            h = (1.0 - z) * n + z * h
            out.append(h)

        return torch.stack(out, dim=0)  # (T,B,H)

    def forward(self, input, x_p_0=None, h_0=None, h_p_0=None, dm_nh_0=None, dm_0=None):
        x, x_p_0, h_0, h_p_0, dm_nh_0, dm_0 = self._process_inputs(
            input, x_p_0, h_0, h_p_0, dm_nh_0, dm_0
        )

        # One (or more) stacked layers; each layer consumes the previous layer's output sequence.
        for l in range(self.num_layers):
            # truncate x_p to input feature count
            F = x.size(-1)
            x_p = x_p_0[l][:, :F]     # (B,F)
            h   = h_0[l]              # (B,H)
            h_p = h_p_0[l]            # (B,H)
            dm_nh = dm_nh_0[l]        # (B,H)
            dm    = dm_0[l]           # (B,3H)

            x = self._layer_forward(x, x_p, h, h_p, dm_nh, dm)  # (T,B,H)

        x = x.transpose(0, 1)  # (B,T,H)
        return x


# ----------------------------
# Standalone wrapper class (similar interface to your NeuralNetwork wrapper)
# ----------------------------

class DeltaGRUNetwork:
    def __init__(self,
                 forward_model: bool = True,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 thx: float = 0.0,
                 thh: float = 0.0,
                 use_tcn_skip: bool = True,
                 seq_len: int = 1024,
                 seq_hop: int = 1024,
                 batch_size: int = 16):
        """
        forward_model=True  : learn input -> output
        forward_model=False : learn output -> input (inverse)

        seq_len/seq_hop: how you chop long captures into windows.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device} device")

        self.device = device
        self.forward_model = forward_model

        self.seq_len = int(seq_len)
        self.seq_hop = int(seq_hop)
        self.batch_size = int(batch_size)

        self.nn_model = DeltaGRUModel(
            hidden_size=hidden_size,
            num_layers=num_layers,
            thx=thx,
            thh=thh,
            use_tcn_skip=use_tcn_skip
        ).to(self.device)

    # --------- helpers: complex -> IQ ---------

    def _to_iq_norm(self, sig_complex: np.ndarray):
        """
        Phase-normalize like your other codebase:
          sig_n = sig * Dataset.conj_phase(sig)
        Returns float32 array (N,2): [I,Q]
        """
        sig_n = sig_complex * Dataset.conj_phase(sig_complex)
        iq = np.stack([np.real(sig_n), np.imag(sig_n)], axis=-1).astype(np.float32)
        return iq

    def _iq_to_complex(self, iq: np.ndarray):
        return iq[..., 0] + 1j * iq[..., 1]

    # --------- data alignment ---------

    def training_data(self, dataset):
        """
        Returns aligned arrays (N,2) float32 for X and Y in normalized domain.
        """
        if self.forward_model:
            x_in, y_out = dataset.input_data, dataset.output_data
        else:
            y_out, x_in = dataset.input_data, dataset.output_data

        X = self._to_iq_norm(x_in)
        Y = self._to_iq_norm(y_out)
        return X, Y

    def _make_sequences(self, X: np.ndarray, Y: np.ndarray):
        """
        X,Y: (N,2)
        Returns: Xs,Ys: (M,T,2)
        """
        if X.shape != Y.shape or X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(f"Expected X,Y shaped (N,2). Got X={X.shape} Y={Y.shape}")

        N = X.shape[0]
        T = self.seq_len
        hop = self.seq_hop
        if N < T:
            raise ValueError(f"Signal too short for seq_len={T}. N={N}")

        starts = range(0, N - T + 1, hop)
        Xs = np.stack([X[s:s+T] for s in starts], axis=0).astype(np.float32)
        Ys = np.stack([Y[s:s+T] for s in starts], axis=0).astype(np.float32)
        return Xs, Ys

    def build_dataloaders(self, X: np.ndarray, Y: np.ndarray, shuffle=True):
        Xs, Ys = self._make_sequences(X, Y)
        Xt = torch.tensor(Xs, dtype=torch.float32)
        Yt = torch.tensor(Ys, dtype=torch.float32)
        ds = TensorDataset(Xt, Yt)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=False)

    # --------- training ---------

    def get_best_model(self, num_epochs, training_dataset, validation_dataset, learning_rate=1e-3):
        """
        Train and restore best weights by validation loss.
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.nn_model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        train_losses, valid_losses = [], []
        best_valid_loss = float("inf")
        best_state = None
        best_epoch = 0

        Xtr, Ytr = self.training_data(training_dataset)
        Xva, Yva = self.training_data(validation_dataset)

        train_loader = self.build_dataloaders(Xtr, Ytr, shuffle=True)
        valid_loader = self.build_dataloaders(Xva, Yva, shuffle=False)

        for epoch in range(num_epochs):
            self.nn_model.train()
            running_train = 0.0

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                preds = self.nn_model(xb)  # (B,T,2)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

                running_train += loss.item() * xb.size(0)

            self.nn_model.eval()
            running_valid = 0.0
            with torch.no_grad():
                for xb, yb in valid_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    preds = self.nn_model(xb)
                    loss = criterion(preds, yb)
                    running_valid += loss.item() * xb.size(0)

            train_loss = running_train
            valid_loss = running_valid
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            scheduler.step(valid_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_state = copy.deepcopy(self.nn_model.state_dict())
                best_epoch = epoch + 1

            if (epoch + 1) % 10 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch+1:3d}/{num_epochs}  Loss={train_loss:.4e}  Valid={valid_loss:.4e}  LR={lr:.2e}")

        if best_state is None:
            raise RuntimeError("Training did not produce a best_state (unexpected).")

        self.nn_model.load_state_dict(best_state)
        print(f"\nBest model from epoch {best_epoch} with validation loss: {best_valid_loss:.4e}")
        return train_losses, valid_losses, best_epoch

    # --------- inference ---------

    def generate_model_output(self, x_complex: np.ndarray):
        """
        Returns complex numpy array prediction (same length as x_complex),
        phase de-normalized using *input* phase reference (consistent with your code style).
        """
        self.nn_model.eval()
        X = self._to_iq_norm(x_complex)  # (N,2)
        N = X.shape[0]

        # If short enough, do one shot; otherwise do sliding chunks and stitch.
        # (This stitching is a simple overlap-add average. If you use hop==len it's exact.)
        T = self.seq_len
        hop = self.seq_hop

        if N < T:
            raise ValueError(f"Input length N={N} is shorter than seq_len={T}")

        preds_acc = np.zeros((N, 2), dtype=np.float64)
        counts = np.zeros((N, 1), dtype=np.float64)

        with torch.no_grad():
            for s in range(0, N - T + 1, hop):
                x_win = torch.tensor(X[s:s+T][None, ...], dtype=torch.float32, device=self.device)  # (1,T,2)
                y_win = self.nn_model(x_win).squeeze(0).cpu().numpy().astype(np.float64)            # (T,2)

                preds_acc[s:s+T] += y_win
                counts[s:s+T] += 1.0

        # Handle tail if (N-T) not aligned: run last window ending at N
        last_start = N - T
        if last_start % hop != 0:
            with torch.no_grad():
                x_win = torch.tensor(X[last_start:last_start+T][None, ...], dtype=torch.float32, device=self.device)
                y_win = self.nn_model(x_win).squeeze(0).cpu().numpy().astype(np.float64)
            preds_acc[last_start:last_start+T] += y_win
            counts[last_start:last_start+T] += 1.0

        preds_iq = (preds_acc / np.maximum(counts, 1.0)).astype(np.float32)  # (N,2)
        y_pred_norm = self._iq_to_complex(preds_iq)

        # Phase de-normalise using input phase reference
        phase = Dataset.conj_phase(x_complex)
        y_pred = y_pred_norm * np.conj(phase)
        return y_pred

    def calculate_forward_nmse(self, dataset):
        if not self.forward_model:
            raise ValueError("Model is not a forward model")
        y_true = dataset.output_data
        y_pred = self.generate_model_output(dataset.input_data)
        nmse = 10 * np.log10(np.sum(np.abs(y_true - y_pred) ** 2) / np.sum(np.abs(y_true) ** 2))
        return nmse
