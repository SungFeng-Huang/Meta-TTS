from .specaug import SpecAug
from .specaug_pp import SpecAugmentation, SpecMixAugmentation, SpecCutAugmentation


__all__ = ["SpecAugment", "SpecAugmentZM", "SpecAugmentMM", "SpecAugmentCM"]


class SpecAugment(SpecAug):
    """ SpecAugment wrapper for input shape resizing. """

    def forward(self, xs, x_lengths=None):
        """ Assert xs padded.
        XvecTDNN input shape: (batch_size, freq, time)
        SpecAug input shape: list of (time, freq)
        apply_specaug input shape: (batch_size, time, freq)
        """
        if self.training is False:
            return xs, x_lengths

        xs_pad = xs.transpose(1, 2).contiguous()
        xs_pad, _ = self.apply_specaug(xs_pad, x_lengths)
        xs_pad = xs_pad.transpose(1, 2).contiguous()

        return xs_pad, x_lengths


class SpecAugmentZM(SpecAugmentation):
    """ SpecAugment++ wrapper for input shape resizing. """

    def forward(self, input):
        """
        XvecTDNN input shape: (batch_size, freq, time)
        SpecAug input shape: (batch_size, channel, time, freq)
        """
        if self.training is False:
            return input

        orig_size = input.shape
        assert len(orig_size) == 3
        x = input.transpose(1, 2).contiguous().unsqueeze(1)
        x = super()(x)
        x = x.squeeze(1).transpose(1, 2).contiguous()
        assert x.shape == orig_size
        return x


class SpecAugmentMM(SpecMixAugmentation):
    """ SpecAugment++ wrapper for input shape resizing. """

    def forward(self, input):
        """
        XvecTDNN input shape: (batch_size, freq, time)
        SpecAug input shape: (batch_size, channel, time, freq)
        """
        if self.training is False:
            return input

        orig_size = input.shape
        assert len(orig_size) == 3
        x = input.transpose(1, 2).contiguous().unsqueeze(1)
        x = super()(x)
        x = x.squeeze(1).transpose(1, 2).contiguous()
        assert x.shape == orig_size
        return x


class SpecAugmentCM(SpecCutAugmentation):
    """ SpecAugment++ wrapper for input shape resizing. """

    def forward(self, input):
        """
        XvecTDNN input shape: (batch_size, freq, time)
        SpecAug input shape: (batch_size, channel, time, freq)
        """
        if self.training is False:
            return input

        orig_size = input.shape
        assert len(orig_size) == 3
        x = input.transpose(1, 2).contiguous().unsqueeze(1)
        x = super()(x)
        x = x.squeeze(1).transpose(1, 2).contiguous()
        assert x.shape == orig_size
        return x
