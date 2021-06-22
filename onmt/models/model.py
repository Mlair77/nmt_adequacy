""" Onmt NMT Model base class definition """
import torch.nn as nn
from onmt.modules.sparse_activations import LogSparsemax


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False, train_lm=False, generator=None,lm_embeddings=None, \
        report_lm=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        # we don't need token drop
        # if not train_lm:  
        #     tgt_0 = tgt[1:]  # to compute language model pred_prob
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder(src, lengths)

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        if train_lm and self.training:   # train NMT and LM model together
            dec_out, attns , lm_outs = self.decoder(tgt, memory_bank,
                                        memory_lengths=lengths)
        elif self.training or report_lm: # train nmt model or valid nmt with nnt_lm
            if generator is None: 
                # to provide generator when token drop/swap
                generator = self.generator[0] if isinstance(self.generator[-1], LogSparsemax) else self.generator
            # dec_out, attns , lm_outs = self.decoder(tgt, memory_bank,
            #                             memory_lengths=lengths, generator=generator, tgt_0=None,
            #                             lm_embeddings=lm_embeddings, report_lm=report_lm)
            output = self.decoder(tgt, memory_bank,
                                        memory_lengths=lengths, generator=generator, tgt_0=None,
                                        lm_embeddings=lm_embeddings, report_lm=report_lm)
            if len(output) == 3:
                dec_out, attns , lm_outs = output
            else:  # model is not 
                dec_out, attns = output
                lm_outs = None
        else: # valid or test
            dec_out, attns = self.decoder(tgt, memory_bank, memory_lengths=lengths)
            return dec_out, attns
        return dec_out, attns, lm_outs

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
