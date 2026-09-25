"""
flsim.airsfl: AirSFL — Split Federated Learning with Zero-Forcing reception and
AirComp aggregation (WCNC methods draft + evaluation roadmap).

Subsystem for the AirSFL study, kept separate from the compute-based flsim core
because its PRIMARY metric is modeled UPLINK COMMUNICATION TIME (not compute
time), and it adds genuinely new radio physics (multi-antenna ZF activation
recovery + AirComp model aggregation) the rest of flsim does not model.

  timing.py — OFDM analog/digital uplink communication-time model (Table I).
  (later) radio.py — MIMO Rayleigh channels, ZF activation recovery + noise,
                     AirComp aggregation + noise (Eq. 9-20).
  (later) model.py — CIFAR ResNet-18 (GroupNorm, 3x3 stem, no maxpool) + cuts.
"""
