"""The decision rule of the multi-axis protocol (paper Section "A Multi-Axis Evaluation Protocol").
Thresholds are conventions fixed here before the validation conditions of run_protocol_validation.ps1 were run
(see protocol_frozen.txt for the hash and the predictions written down in advance)."""
HEADROOM_MIN = 0.05      # no headroom if p0 - c is below this
RECOVERABLE_BELOW = 10   # percent: selected reduction under this -> recoverable
NOT_DECODED_FROM = 90    # percent: selected reduction at or above this -> not decoded by the tested probes
CHANCE_BAND = 0.05       # a discriminator within this of chance (balanced) or of the majority rate (raw) "reads chance"


def verdict(rho, headroom):
    if headroom < HEADROOM_MIN:
        return 'no headroom'
    return 'recoverable' if rho < RECOVERABLE_BELOW else ('partly reduced' if rho < NOT_DECODED_FROM else 'not decoded by the tested probes')
