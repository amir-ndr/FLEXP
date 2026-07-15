"""
interfaces/splittable.py: Contract for models usable with split learning
(flsim.system.split_model.split_model, flsim.core.split_simulator.SplitSimulator).

A model supports split learning if it exposes its forward pass as an ORDERED,
flat sequence of layers — i.e. calling each returned layer in order, feeding
each one's output to the next, reproduces forward(). This is what lets
split_model() cut the network at an arbitrary integer index and hand the
client the first `cut_layer` layers, the server the rest, with no other
change to how the layers themselves work.

This is a plain mixin, NOT an abc.ABC — deliberately, to avoid mixing
ABCMeta with torch.nn.Module's own metaclass machinery. split_model() checks
for the method via hasattr(), not isinstance().

Making an existing nn.Module splittable requires ONE additive method; it does
not change forward(), state_dict(), or anything about how the model behaves
in the standard (non-split) sync/async/OTA algorithms already in this
framework — a model can be used by both at once.

Example:
    class MyCNN(nn.Module, Splittable):
        def __init__(self):
            super().__init__()
            self.features   = nn.Sequential(nn.Conv2d(...), nn.ReLU(), ...)
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(...), ...)

        def forward(self, x):
            return self.classifier(self.features(x))

        def ordered_layers(self) -> list:
            return list(self.features) + list(self.classifier)
"""


class Splittable:
    """Mixin marking a model as usable with split learning."""

    def ordered_layers(self) -> list:
        """
        Return this model's layers as a flat, ordered list of nn.Module,
        in the exact order forward() applies them. The list boundaries are
        where split_model() is allowed to cut the network.

        Returns:
            list[nn.Module]: layers in forward-pass order.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement ordered_layers() to be used "
            f"with split learning (see flsim.interfaces.splittable.Splittable)."
        )

    def num_splittable_layers(self) -> int:
        """Convenience: len(self.ordered_layers()). Used to validate cut_layer."""
        return len(self.ordered_layers())
