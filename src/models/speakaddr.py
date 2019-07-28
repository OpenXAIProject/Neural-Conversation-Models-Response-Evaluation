import torch
import torch.nn as nn
import layers
from utils import to_var, pad


class SpeakAddr(nn.Module):
    def __init__(self, config):
        super(SpeakAddr, self).__init__()

        self.config = config
        self.encoder = layers.EncoderRNN(config.vocab_size, config.embedding_size, config.encoder_hidden_size,
                                         nn.LSTM, config.num_layers, config.bidirectional, config.dropout,
                                         pretrained_wv_path=config.pretrained_wv_path)

        self.decoder = layers.DecoderSARNN(config.vocab_size, config.user_size, config.embedding_size,
                                           config.decoder_hidden_size, config.num_layers,
                                           config.dropout, config.word_drop, config.max_unroll, config.sample,
                                           config.temperature, config.beam_size)

        context_input_size = config.encoder_hidden_size * self.encoder.num_directions
        self.context2decoder_h = layers.FeedForward(context_input_size, config.decoder_hidden_size,
                                                    num_layers=1, activation=config.activation)
        self.context2decoder_c = layers.FeedForward(context_input_size, config.decoder_hidden_size,
                                                    num_layers=1, activation=config.activation)

        if config.tie_embedding:
            self.decoder.embedding = self.encoder.embedding

    def forward(self, input_utterances, conv_users, input_utterance_length, input_conversation_length,
                target_utterances, decode=False):
        num_utterances = input_utterances.size(0)
        max_conv_len = input_conversation_length.data.max().item()

        _, encoder_hidden = self.encoder(input_utterances, input_utterance_length)
        start = torch.cumsum(torch.cat((to_var(input_conversation_length.data.new(1).zero_()),
                                        input_conversation_length[:-1])), 0)
        encoder_hidden_h, encoder_hidden_c = encoder_hidden

        encoder_hidden_h = encoder_hidden_h.transpose(1, 0).contiguous().view(num_utterances, -1)
        encoder_hidden_h = torch.stack([pad(encoder_hidden_h.narrow(0, s, l), max_conv_len)
                                        for s, l in zip(start.data.tolist(),
                                                        input_conversation_length.data.tolist())], 0)
        decoder_init_h = self.context2decoder_h(encoder_hidden_h)
        decoder_init_h = decoder_init_h.view(-1, self.decoder.hidden_size)

        encoder_hidden_c = encoder_hidden_c.transpose(1, 0).contiguous().view(num_utterances, -1)
        encoder_hidden_c = torch.stack([pad(encoder_hidden_c.narrow(0, s, l), max_conv_len)
                                        for s, l in zip(start.data.tolist(),
                                                        input_conversation_length.data.tolist())], 0)
        decoder_init_c = self.context2decoder_c(encoder_hidden_c)
        decoder_init_c = decoder_init_c.view(-1, self.decoder.hidden_size)

        decoder_init = (decoder_init_h, decoder_init_c)

        if not decode:
            decoder_outputs = self.decoder(target_utterances, conv_users, init_h=decoder_init, decode=decode)
            return decoder_outputs

        else:
            prediction, final_score, length = self.decoder.beam_decode(init_h=decoder_init, user_inputs=conv_users)

            return prediction

    def generate(self, context, conv_users, utterances_length, n_context):
        samples = []
        all_samples = list()

        for i in range(n_context):
            _, encoder_hidden = self.encoder(context[:, i, :], utterances_length[:, i])

        encoder_hidden_h, encoder_hidden_c = encoder_hidden
        decoder_init_h = self.context2decoder_h(encoder_hidden_h)
        decoder_init_h = decoder_init_h.view(-1, self.decoder.hidden_size)
        decoder_init_c = self.context2decoder_c(encoder_hidden_c)
        decoder_init_c = decoder_init_c.view(-1, self.decoder.hidden_size)

        decoder_init = (decoder_init_h, decoder_init_c)

        prediction_all, final_score, length = self.decoder.beam_decode(init_h=decoder_init, user_inputs=conv_users)
        all_samples.append(prediction_all)
        prediction = prediction_all[:, 0, :]
        samples.append(prediction)

        samples = torch.stack(samples, 1)
        all_samples = torch.stack(all_samples, 1)

        return samples, all_samples
