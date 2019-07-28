import torch
import torch.nn as nn
import layers


class SpeakAddr(nn.Module):
    def __init__(self, config):
        super(SpeakAddr, self).__init__()

        self.config = config
        self.encoder = layers.EncoderRNN(config.vocab_size, config.user_size, config.embedding_size,
                                         config.encoder_hidden_size, config.pretrained_wv_path)

        self.decoder = layers.DecoderSARNN(config.vocab_size, config.embedding_size, config.decoder_hidden_size,
                                           config.rnncell, config.num_layers, config.dropout, config.word_drop,
                                           config.max_unroll, config.sample, config.temperature, config.beam_size)

        self.context2decoder = layers.FeedForward(config.encoder_hidden_size,
                                                  config.num_layers * config.decoder_hidden_size,
                                                  num_layers=1,
                                                  activation=config.activation)

        if config.tie_embedding:
            self.decoder.embedding = self.encoder.embedding

    def forward(self, input_sentences, input_users, input_sentence_length, input_conversation_length, target_sentences,
                decode=False):
        encoder_outputs, encoder_hidden = self.encoder(input_sentences, input_sentence_length)

        context_outputs = torch.cat([encoder_outputs[i, :l, :] for i, l in enumerate(input_conversation_length.data)])

        decoder_init = self.context2decoder(context_outputs)
        decoder_init = decoder_init.view(self.decoder.num_layers, -1, self.decoder.hidden_size)

        if not decode:
            decoder_outputs = self.decoder(target_sentences, input_users, init_h=decoder_init, decode=decode)
            return decoder_outputs

        else:
            prediction, final_score, length = self.decoder.beam_decode(init_h=decoder_init, user_inputs=input_users)

            return prediction

    def generate(self, context, conv_users, sentence_length, n_context):
        samples = []
        all_samples = list()

        context_outputs = None
        for i in range(n_context):
            context_outputs, encoder_hidden = self.encoder(context[:, i, :], sentence_length[:, i])

        decoder_init = self.context2decoder(context_outputs)
        decoder_init = decoder_init.view(self.decoder.num_layers, -1, self.decoder.hidden_size)

        prediction_all, final_score, length = self.decoder.beam_decode(init_h=decoder_init, user_inputs=conv_users)
        all_samples.append(prediction_all)
        prediction = prediction_all[:, 0, :]
        samples.append(prediction)

        samples = torch.stack(samples, 1)
        all_samples = torch.stack(all_samples, 1)

        return samples, all_samples
