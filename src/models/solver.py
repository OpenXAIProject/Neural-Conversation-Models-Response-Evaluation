import numpy as np
import torch
import torch.nn as nn
import models
from layers import masked_cross_entropy
from utils import to_var, TensorboardWriter
import os
from tqdm import tqdm
from math import isnan
import re
import codecs


class Solver(object):
    def __init__(self, config, train_data_loader, eval_data_loader, vocab, is_train=True, model=None):
        self.config = config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.vocab = vocab
        self.is_train = is_train
        self.model = model
        self.writer = None
        self.optimizer = None
        self.epoch_loss = None
        self.validation_loss = None

    def build(self, cuda=True):
        if self.model is None:
            self.model = getattr(models, self.config.model)(self.config)

            if self.config.mode == 'train' and self.config.checkpoint is None:
                print('Parameter initiailization')
                for name, param in self.model.named_parameters():
                    if 'weight_hh' in name:
                        print('\t' + name)
                        nn.init.orthogonal_(param)

                    if 'bias_hh' in name:
                        print('\t' + name)
                        dim = int(param.size(0) / 3)
                        param.data[dim:2 * dim].fill_(2.0)

        if torch.cuda.is_available() and cuda:
            self.model.cuda()

        if self.config.checkpoint:
            self.load_model(self.config.checkpoint)

        if self.is_train:
            self.writer = TensorboardWriter(self.config.logdir)
            self.optimizer = self.config.optimizer(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                   lr=self.config.learning_rate)

    def save_model(self, epoch):
        ckpt_path = os.path.join(self.config.save_path, f'{epoch}.pkl')
        print(f'Save parameters to {ckpt_path}')
        torch.save(self.model.state_dict(), ckpt_path)

    def load_model(self, checkpoint):
        print(f'Load parameters from {checkpoint}')
        epoch = re.match(r"[0-9]*", os.path.basename(checkpoint)).group(0)
        self.epoch_i = int(epoch)
        self.model.load_state_dict(torch.load(checkpoint))

    def write_summary(self, epoch_i):
        epoch_loss = getattr(self, 'epoch_loss', None)
        if epoch_loss is not None:
            self.writer.update_loss(loss=epoch_loss, step_i=epoch_i + 1, name='train_loss')

        epoch_recon_loss = getattr(self, 'epoch_recon_loss', None)
        if epoch_recon_loss is not None:
            self.writer.update_loss(loss=epoch_recon_loss, step_i=epoch_i + 1, name='train_recon_loss')

        epoch_kl_div = getattr(self, 'epoch_kl_div', None)
        if epoch_kl_div is not None:
            self.writer.update_loss(loss=epoch_kl_div, step_i=epoch_i + 1, name='train_kl_div')

        kl_mult = getattr(self, 'kl_mult', None)
        if kl_mult is not None:
            self.writer.update_loss(loss=kl_mult, step_i=epoch_i + 1, name='kl_mult')

        epoch_bow_loss = getattr(self, 'epoch_bow_loss', None)
        if epoch_bow_loss is not None:
            self.writer.update_loss(loss=epoch_bow_loss, step_i=epoch_i + 1, name='bow_loss')

        validation_loss = getattr(self, 'validation_loss', None)
        if validation_loss is not None:
            self.writer.update_loss(loss=validation_loss, step_i=epoch_i + 1, name='validation_loss')

    def train(self):
        epoch_loss_history = list()
        for epoch_i in range(self.epoch_i, self.config.n_epoch):
            self.epoch_i = epoch_i
            batch_loss_history = list()
            self.model.train()
            n_total_words = 0
            for batch_i, (conversations, convs_length, utterances_length) in \
                    enumerate(tqdm(self.train_data_loader, ncols=80)):
                # conversations: [batch_size, max_conv_len, max_utter_len] list of conversation
                #   A conversation: [max_conv_len, max_utter_len] list of utterances
                #   An utterance: [max_utter_len] list of tokens
                # convs_length: [batch_size] list of integer
                # utterances_length: [batch_size, max_conv_len] list of conversation that has a list of utterance length

                input_conversations = [conv[:-1] for conv in conversations]
                target_conversations = [conv[1:] for conv in conversations]

                input_utterances = [utter for conv in input_conversations for utter in conv]
                target_utterances = [utter for conv in target_conversations for utter in conv]
                input_utterance_length = [l for len_list in utterances_length for l in len_list[:-1]]
                target_utterance_length = [l for len_list in utterances_length for l in len_list[1:]]
                input_conversation_length = [conv_len - 1 for conv_len in convs_length]

                input_utterances = to_var(torch.LongTensor(input_utterances))
                target_utterances = to_var(torch.LongTensor(target_utterances))
                input_utterance_length = to_var(torch.LongTensor(input_utterance_length))
                target_utterance_length = to_var(torch.LongTensor(target_utterance_length))
                input_conversation_length = to_var(torch.LongTensor(input_conversation_length))

                self.optimizer.zero_grad()

                utterances_logits = self.model(input_utterances, input_utterance_length,
                                             input_conversation_length, target_utterances, decode=False)

                batch_loss, n_words = masked_cross_entropy(utterances_logits, target_utterances, target_utterance_length)

                assert not isnan(batch_loss.item())
                batch_loss_history.append(batch_loss.item())
                n_total_words += n_words.item()

                if batch_i % self.config.print_every == 0:
                    tqdm.write(f'Epoch: {epoch_i+1}, iter {batch_i}: loss = {batch_loss.item()/ n_words.item():.3f}')

                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.optimizer.step()

            epoch_loss = np.sum(batch_loss_history) / n_total_words
            epoch_loss_history.append(epoch_loss)
            self.epoch_loss = epoch_loss

            print(f'Epoch {epoch_i+1} loss average: {epoch_loss:.3f}')

            if epoch_i % self.config.save_every_epoch == 0:
                self.save_model(epoch_i + 1)

            print('\n<Validation>...')
            self.validation_loss = self.evaluate()

            if epoch_i % self.config.plot_every_epoch == 0:
                self.write_summary(epoch_i)

        self.save_model(self.config.n_epoch)

        return epoch_loss_history

    def evaluate(self):
        self.model.eval()
        batch_loss_history = []
        n_total_words = 0
        for batch_i, (conversations, convs_length, utterances_length) in \
                enumerate(tqdm(self.eval_data_loader, ncols=80)):
            input_conversations = [conv[:-1] for conv in conversations]
            target_conversations = [conv[1:] for conv in conversations]

            input_utterances = [utter for conv in input_conversations for utter in conv]
            target_utterances = [utter for conv in target_conversations for utter in conv]
            input_utterance_length = [l for len_list in utterances_length for l in len_list[:-1]]
            target_utterance_length = [l for len_list in utterances_length for l in len_list[1:]]
            input_conversation_length = [conv_len - 1 for conv_len in convs_length]

            with torch.no_grad():
                input_utterances = to_var(torch.LongTensor(input_utterances))
                target_utterances = to_var(torch.LongTensor(target_utterances))
                input_utterance_length = to_var(torch.LongTensor(input_utterance_length))
                target_utterance_length = to_var(torch.LongTensor(target_utterance_length))
                input_conversation_length = to_var(torch.LongTensor(input_conversation_length))

            utterances_logits = self.model(input_utterances, input_utterance_length,
                                         input_conversation_length, target_utterances)

            batch_loss, n_words = masked_cross_entropy(utterances_logits, target_utterances, target_utterance_length)

            assert not isnan(batch_loss.item())
            batch_loss_history.append(batch_loss.item())
            n_total_words += n_words.item()

        epoch_loss = np.sum(batch_loss_history) / n_total_words

        print(f'Validation loss: {epoch_loss:.3f}\n')

        return epoch_loss

    def test(self):
        self.model.eval()
        batch_loss_history = []
        n_total_words = 0
        for batch_i, (conversations, convs_length, utterances_length) in \
                enumerate(tqdm(self.eval_data_loader, ncols=80)):
            input_conversations = [conv[:-1] for conv in conversations]
            target_conversations = [conv[1:] for conv in conversations]

            input_utterances = [utter for conv in input_conversations for utter in conv]
            target_utterances = [utter for conv in target_conversations for utter in conv]
            input_utterance_length = [l for len_list in utterances_length for l in len_list[:-1]]
            target_utterance_length = [l for len_list in utterances_length for l in len_list[1:]]
            input_conversation_length = [conv_len - 1 for conv_len in convs_length]

            with torch.no_grad():
                input_utterances = to_var(torch.LongTensor(input_utterances))
                target_utterances = to_var(torch.LongTensor(target_utterances))
                input_utterance_length = to_var(torch.LongTensor(input_utterance_length))
                target_utterance_length = to_var(torch.LongTensor(target_utterance_length))
                input_conversation_length = to_var(torch.LongTensor(input_conversation_length))

            utterances_logits = self.model(input_utterances, input_utterance_length,
                                         input_conversation_length, target_utterances)

            batch_loss, n_words = masked_cross_entropy(utterances_logits, target_utterances, target_utterance_length)

            assert not isnan(batch_loss.item())
            batch_loss_history.append(batch_loss.item())
            n_total_words += n_words.item()

        epoch_loss = np.sum(batch_loss_history) / n_total_words

        print(f'Number of words: {n_total_words}')
        print(f'Bits per word: {epoch_loss:.3f}')
        word_perplexity = np.exp(epoch_loss)
        print(f'Word perplexity : {word_perplexity:.3f}\n')

        return word_perplexity

    def export_samples(self, beam_size=5):
        self.model.decoder.beam_size = beam_size
        self.model.eval()
        n_context = self.config.n_context
        n_sample_step = self.config.n_sample_step
        context_history = list()
        sample_history = list()
        ground_truth_history = list()
        for batch_i, (conversations, convs_length, utterances_length) in \
                enumerate(tqdm(self.eval_data_loader, ncols=80)):
            conv_indices = [i for i in range(len(conversations)) if len(conversations[i]) >= n_context + n_sample_step]
            context = [c for i in conv_indices for c in [conversations[i][:n_context]]]
            ground_truth = [c for i in conv_indices for c in [conversations[i][n_context:n_context + n_sample_step]]]
            utterances_length = [c for i in conv_indices for c in [utterances_length[i][:n_context]]]

            with torch.no_grad():
                context = to_var(torch.LongTensor(context))
                utterances_length = to_var(torch.LongTensor(utterances_length))

            _, all_samples = self.model.generate(context, utterances_length, n_context)

            context = context.data.cpu().numpy().tolist()
            all_samples = all_samples.data.cpu().numpy().tolist()
            context_history.append(context)
            sample_history.append(all_samples)
            ground_truth_history.append(ground_truth)

        target_file_name = 'responses_{}_{}_{}_{}_{}.txt'.format(self.config.mode, n_context, n_sample_step,
                                                                 beam_size, self.epoch_i)
        print("Writing candidates into file {}".format(target_file_name))
        conv_idx = 0
        with codecs.open(os.path.join(self.config.save_path, target_file_name), 'w', "utf-8") as output_f:
            for contexts, samples, ground_truths in tqdm(zip(context_history, sample_history, ground_truth_history),
                                                         total=len(context_history), ncols=80):
                for one_conv_contexts, one_conv_samples, one_conv_ground_truth in zip(contexts, samples, ground_truths):
                    print("Conversation Context {}".format(conv_idx), file=output_f)
                    print("\n".join([self.vocab.decode(utter) for utter in one_conv_contexts]), file=output_f)
                    print("\n".join([self.vocab.decode(utter) for utters_beam in one_conv_samples for utter in utters_beam]), file=output_f)
                    print("\n".join([self.vocab.decode(utter) for utter in one_conv_ground_truth]), file=output_f)
                    conv_idx += 1

        return conv_idx
