"""
core/simulator.py: Main orchestrator for the federated learning simulation loop.

Simulator wires all injected components together and drives the round loop.
It is the only core object that calls methods across multiple interfaces.

Component responsibilities:
  Server      — holds global model, delegates to algorithm
  Client      — local training (PyTorch)
  TimeModel   — simulated time computation (analytic formula)
  EnergyModel — energy computation (analytic formula)
  ChannelModel — channel gain and bandwidth allocation
  Evaluator   — test-set evaluation
  Logger      — CSV logging and plotting

IMPORTANT: Simulator never instantiates any component — all dependencies are
injected by run.py, which is the only file allowed to import from all layers.
"""

import copy

import torch
import numpy as np

from flsim.core.logger import RoundResult
from flsim.system.energy import EnergyModel


class Simulator:
    """
    Federated learning simulation orchestrator.

    Drives the global round loop, wires client training with simulated time
    and energy accounting, and calls the logger at each round.

    This class does NOT:
    - Instantiate any component (all are injected).
    - Import any concrete implementation (algorithm, channel model, etc.).
    - Perform dataset loading or model creation.
    """

    def __init__(
        self,
        server,
        clients: list,
        time_model,
        channel_model,
        allocator,
        energy_model: EnergyModel,
        evaluator,
        logger,
        config,
        rng: np.random.RandomState,
        device: torch.device,
    ):
        """
        Args:
            server: Server object with global_model and algorithm.
            clients: list of Client objects (all K clients).
            time_model: TimeModel implementation.
            channel_model: ChannelModel implementation.
            allocator: ResourceAllocator implementation.
            energy_model: EnergyModel instance.
            evaluator: Evaluator instance.
            logger: Logger instance.
            config: SimpleNamespace/dict-like config object from run.py.
            rng: seeded numpy RandomState for reproducibility.
            device: torch.device for training.
        """
        self.server = server
        self.clients = clients
        self.time_model = time_model
        self.channel_model = channel_model
        self.allocator = allocator
        self.energy_model = energy_model
        self.evaluator = evaluator
        self.logger = logger
        self.config = config
        self.rng = rng
        self.device = device

        # Resolve upload size once at startup, not per round.
        self._upload_bits = _resolve_upload_bits(
            config.wireless, server.global_model
        )
        # Cache pre-converted scalar constants so _run_round stays cross-layer-free.
        self._p_max_w  = config._p_max_w
        self._f_max_hz = config._f_max_hz
        # If downlink_negligible is set, the model-broadcast (download) time is
        # treated as 0 (base station has effectively unlimited power/bandwidth).
        self._downlink_negligible = bool(
            getattr(config.wireless, "downlink_negligible", False)
        )

    def run(self) -> None:
        """
        Execute the full federated learning experiment.

        Stopping condition (mutually exclusive):
          - Default: run exactly learning.global_rounds communication rounds.
          - If learning.stop_by_time_s is set (> 0): ignore global_rounds and
            instead run rounds until cumulative simulated_time_s reaches that
            budget. See _should_continue().

        After every evaluate_every rounds, evaluate the global model on the
        test set. At the end, generate result plots.
        """
        simulated_time_s = 0.0
        cfg_learn = self.config.learning
        cfg_eval  = self.config.evaluation

        stop_by_time_s = getattr(cfg_learn, "stop_by_time_s", None)
        time_based = stop_by_time_s is not None and stop_by_time_s > 0

        cfg_w = self.config.wireless
        if time_based:
            print(f"\n[Simulator] Starting: stop_by_time_s={stop_by_time_s:.0f}s (time-based), "
                  f"{cfg_learn.clients_per_round}/{len(self.clients)} clients/round, "
                  f"device={self.device}")
        else:
            print(f"\n[Simulator] Starting: {cfg_learn.global_rounds} rounds, "
                  f"{cfg_learn.clients_per_round}/{len(self.clients)} clients/round, "
                  f"device={self.device}")
        print(f"[Simulator] Upload size: {self._upload_bits/1e3:.1f} kbits "
              f"(mode={cfg_w.upload_size_mode})")

        round_idx = 0
        while self._should_continue(round_idx, simulated_time_s, cfg_learn.global_rounds,
                                     time_based, stop_by_time_s):
            round_result = self._run_round(round_idx)
            simulated_time_s += round_result.round_duration_s

            # Evaluate every evaluate_every rounds (and always on round 0)
            eval_result = None
            if round_idx % cfg_eval.evaluate_every == 0:
                eval_result = self.evaluator.evaluate(
                    self.server.global_model, device=self.device
                )

            self.logger.log_round(round_idx, simulated_time_s, round_result, eval_result)

            if eval_result is not None:
                print(
                    f"  Round {round_idx:4d} | "
                    f"sim_time={simulated_time_s:.1f}s | "
                    f"acc={eval_result.test_accuracy:.4f} | "
                    f"loss={eval_result.test_loss:.4f} | "
                    f"round_dur={round_result.round_duration_s:.2f}s"
                )

            round_idx += 1

        print("[Simulator] Done. Saving plots …")
        self.logger.plot_results()

    @staticmethod
    def _should_continue(round_idx: int, simulated_time_s: float, global_rounds: int,
                          time_based: bool, stop_by_time_s: float) -> bool:
        """Round-based: round_idx < global_rounds. Time-based: simulated_time_s < stop_by_time_s."""
        if time_based:
            return simulated_time_s < stop_by_time_s
        return round_idx < global_rounds

    def _run_round(self, round_idx: int) -> RoundResult:
        """
        Execute one communication round.

        Steps:
          1. Select clients via algorithm.
          2. Allocate bandwidth via channel model.
          3. Compute channel gains.
          4. Each client trains locally (real PyTorch — wall-clock time NOT recorded).
          5. TimeModel computes simulated durations analytically.
          6. EnergyModel computes energy consumed.
          7. Server aggregates.
          8. Build and return RoundResult.

        Args:
            round_idx (int): current round index.

        Returns:
            RoundResult with all system metrics for this round.
        """
        cfg = self.config.learning
        cfg_w = self.config.wireless

        # --- 1. Select clients ---
        # Pass system context so channel-aware selectors can rank clients
        # before selection (same contract as the async simulator).
        selection_ctx = {
            "channel_model":      self.channel_model,
            "noise_psd_w_per_hz": self.config._noise_psd_w_per_hz,
            "bw_per_client_hz":   cfg_w.total_bandwidth_hz / max(1, cfg.clients_per_round),
            "upload_size_bits":   self._upload_bits,
            "round_idx":          round_idx,
        }
        selected = self.server.select_clients(
            self.clients, cfg.clients_per_round, self.rng, **selection_ctx
        )
        selected_profiles = [c.profile for c in selected]
        selected_ids = [c.client_id for c in selected]

        # --- 2. Compute channel gains (needed before allocation so allocators
        #        can use them for channel-aware decisions) ---
        gains = {
            c.client_id: self.channel_model.channel_gain(c.profile, self.rng)
            for c in selected
        }

        # --- 3. Resource allocation ---
        noise_psd = self.config._noise_psd_w_per_hz

        bw_alloc = self.allocator.allocate_bandwidth(
            selected_profiles, cfg_w.total_bandwidth_hz,
            channel_gains=gains,
        )
        pw_alloc = self.allocator.allocate_power(
            selected_profiles, self._p_max_w,
            channel_gains=gains,
        )
        fq_alloc = self.allocator.allocate_cpu_freq(
            selected_profiles, self._f_max_hz,
            channel_gains=gains,
        )

        upload_bits = self._upload_bits

        # --- 4. Local training + 5. Simulated time + 6. Energy ---
        client_updates = []
        compute_times, upload_times, download_times = [], [], []
        compute_energies, tx_energies = [], []
        rates_list = []

        for client in selected:
            cid   = client.client_id
            bw_hz = bw_alloc[cid]
            p_w   = pw_alloc[cid]
            f_hz  = fq_alloc[cid]
            g_k   = gains[cid]

            # Compute achievable rate for this client
            rate_bps = self.channel_model.achievable_rate_bps(
                bandwidth_hz=bw_hz,
                tx_power_w=p_w,
                channel_gain=g_k,
                noise_psd_w_per_hz=noise_psd,
            )

            # Simulated time — use allocator-assigned f_hz and g_k from this round.
            t_comp = self.time_model.compute_training_time(
                client.profile,
                num_samples=client.num_samples,
                local_epochs=cfg.local_epochs,
                batch_size=cfg.batch_size,
                cpu_freq_hz=f_hz,
            )
            # Pass g_k so the time model uses the same fading draw as above —
            # critical for ExpFadingChannelModel where ρ must not be re-drawn.
            t_up = self.time_model.compute_upload_time(
                client.profile,
                size_bits=upload_bits,
                bandwidth_hz=bw_hz,
                channel_gain=g_k,
            )
            t_dn = 0.0 if self._downlink_negligible else \
                self.time_model.compute_download_time(
                    client.profile,
                    size_bits=upload_bits,
                    bandwidth_hz=bw_hz,
                    channel_gain=g_k,
                )

            # Energy — use allocator-assigned f_hz and p_w.
            e_comp = self.energy_model.compute_energy_j(
                client.profile,
                local_epochs=cfg.local_epochs,
                num_samples=client.num_samples,
                cpu_freq_hz=f_hz,
            )
            e_tx = self.energy_model.transmission_energy_j(
                client.profile, upload_time_s=t_up, tx_power_w=p_w,
            )

            # PyTorch local training (wall-clock time is NOT simulated time)
            self.server.algorithm.configure_client(
                client, self.server.global_model, round_idx
            )
            state_dict, n_samples, train_loss = client.train(
                global_model=self.server.global_model,
                local_epochs=cfg.local_epochs,
                batch_size=cfg.batch_size,
                learning_rate=cfg.learning_rate,
                device=self.device,
                proximal_mu=getattr(client, "proximal_mu", 0.0),
            )

            from flsim.core.client import ClientUpdate
            update = ClientUpdate(
                client_id=cid,
                state_dict=state_dict,
                num_samples=n_samples,
                train_loss=train_loss,
                compute_time_s=t_comp,
                upload_time_s=t_up,
                download_time_s=t_dn,
                total_time_s=t_comp + t_up + t_dn,
                compute_energy_j=e_comp,
                tx_energy_j=e_tx,
                total_energy_j=e_comp + e_tx,
                channel_gain=g_k,
                achievable_rate_bps=rate_bps,
                allocated_bandwidth_hz=bw_hz,
            )
            client_updates.append(update)
            compute_times.append(t_comp)
            upload_times.append(t_up)
            download_times.append(t_dn)
            compute_energies.append(e_comp)
            tx_energies.append(e_tx)
            rates_list.append(rate_bps)

        # --- 7. Server aggregates ---
        self.server.aggregate(client_updates)

        # Synchronous round: bottlenecked by the slowest client
        round_duration = max(
            u.total_time_s for u in client_updates
        )

        # --- 8. Build RoundResult ---
        return RoundResult(
            round_idx=round_idx,
            round_duration_s=round_duration,
            compute_times_s=compute_times,
            upload_times_s=upload_times,
            download_times_s=download_times,
            compute_energies_j=compute_energies,
            tx_energies_j=tx_energies,
            channel_gains=list(gains.values()),
            rates_bps=rates_list,
            selected_client_ids=selected_ids,
            train_losses=[u.train_loss for u in client_updates],
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _resolve_upload_bits(cfg_w, model) -> float:
    """
    Determine the upload size in bits from config.

    Two modes (set via wireless.upload_size_mode):

      "fixed"  — use wireless.upload_size_bits directly.
                 Good for comparing against paper numbers (e.g. 28,100 bits).

      "model"  — compute from the model's full state_dict:
                 bits = (total float32 elements) × 32 bits.
                 Includes parameters AND buffers (e.g. BatchNorm running
                 mean/var), which is what is actually transmitted in FedAvg.
                 This updates automatically if the architecture changes.

    Args:
        cfg_w: wireless config namespace with upload_size_mode and upload_size_bits.
        model: nn.Module whose state_dict defines the transmitted payload.

    Returns:
        float: upload size in bits.

    Raises:
        ValueError: if upload_size_mode is not "fixed" or "model".
    """
    mode = cfg_w.upload_size_mode
    if mode == "fixed":
        return float(cfg_w.upload_size_bits)
    elif mode == "model":
        # Count every element in the state_dict (params + buffers)
        total_elements = sum(t.numel() for t in model.state_dict().values())
        bits = total_elements * 32   # float32 = 32 bits per element
        return float(bits)
    else:
        raise ValueError(
            f"Unknown upload_size_mode '{mode}'. Choose 'fixed' or 'model'."
        )
