from utils.vocab import EOS_TOKEN, PAD_TOKEN


def pad_utters_users(conversations, max_utters_length, max_conversation_length):
    def pad_tokens(tokens):
        n_valid_tokens = len(tokens)
        if n_valid_tokens > max_utters_length - 1:
            tokens = tokens[:max_utters_length - 1]
        n_pad = max_utters_length - n_valid_tokens - 1
        tokens = tokens + [EOS_TOKEN] + [PAD_TOKEN] * n_pad
        return tokens

    def pad_conversation(one_conversation):
        return [pad_tokens(utter) for utter in one_conversation]

    all_padded_utters = list()
    all_utter_length = list()

    for conversation in conversations:
        if len(conversation) > max_conversation_length:
            conversation = conversation[:max_conversation_length]
        utter_length = [min(len(utter) + 1, max_utters_length) for utter in conversation]
        all_utter_length.append(utter_length)

        utters = pad_conversation(conversation)
        all_padded_utters.append(utters)

    return all_padded_utters, all_utter_length
