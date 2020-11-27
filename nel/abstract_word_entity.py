import torch
import torch.nn as nn

import io
from nel.vocabulary import Vocabulary
import json


def load(path, model_class, suffix=''):
    with io.open(path + '.config', 'r', encoding='utf8') as f:
        config = json.load(f)

    word_voca = Vocabulary()
    word_voca.__dict__ = config['word_voca']
    config['word_voca'] = word_voca
    entity_voca = Vocabulary()
    entity_voca.__dict__ = config['entity_voca']
    config['entity_voca'] = entity_voca

    if 'snd_word_voca' in config:
        snd_word_voca = Vocabulary()
        snd_word_voca.__dict__ = config['snd_word_voca']
        config['snd_word_voca'] = snd_word_voca
    if 'aet_word_voca' in config:
        aet_word_voca = Vocabulary()
        aet_word_voca.__dict__ = config['aet_word_voca']
        config['aet_word_voca'] = aet_word_voca

    model = model_class(config)
    model.load_state_dict(torch.load(path + '.state_dict' + suffix))
    return model


class AbstractWordEntity(nn.Module):
    """
    abstract class containing word and entity embeddings and vocabulary
    """

    def __init__(self, config=None):
        super(AbstractWordEntity, self).__init__()
        if config is None:
            return

        self.emb_dims = config['emb_dims']
        self.aet_dims = config['aet_dims']
        self.word_voca = config['word_voca']
        self.entity_voca = config['entity_voca']
        # whether or note freeze the embeddings
        self.freeze_embs = config['freeze_embs']

        self.word_embeddings = config['word_embeddings_class'](self.word_voca.size(), self.emb_dims)
        self.entity_embeddings = config['entity_embeddings_class'](self.entity_voca.size(), self.emb_dims)
        """anonymous entities type words based entity embeddings"""
        self.aet_entity_embeddings = config['entity_embeddings_class'](self.entity_voca.size(), self.aet_dims)

        # load the embedding vectors to Embedding.weight. nn.Parameter is a kind of Tensor that is to be considered a module parameter.
        #
        # nn.Parameters are Tensor subclasses, that have a very special property when used with Module s - when they’re assigned as Module
        # attributes they are automatically added to the list of its parameters, and will appear e.g. in parameters() iterator. Assigning
        # a Tensor doesn’t have such effect. This is because one might want to cache some temporary state, like last hidden state of the
        # RNN, in the model. If there was no such class as Parameter, these temporaries would get registered too.
        if 'word_embeddings' in config:
            self.word_embeddings.weight = nn.Parameter(torch.Tensor(config['word_embeddings']))
        if 'entity_embeddings' in config:
            self.entity_embeddings.weight = nn.Parameter(torch.Tensor(config['entity_embeddings']))
        if 'aet_entity_embeddings' in config:
            self.aet_entity_embeddings.weight = nn.Parameter(torch.Tensor(config['aet_entity_embeddings']))
        # GloVe embeddings
        if 'snd_word_voca' in config:
            self.snd_word_voca = config['snd_word_voca']
            self.snd_word_embeddings = config['word_embeddings_class'](self.snd_word_voca.size(), self.emb_dims)
        if 'snd_word_embeddings' in config:
            self.snd_word_embeddings.weight = nn.Parameter(torch.Tensor(config['snd_word_embeddings']))
        # anonymous entities type words
        if 'aet_word_voca' in config:
            self.aet_word_voca = config['aet_word_voca']
            self.aet_word_embeddings = config['word_embeddings_class'](self.aet_word_voca.size(), self.aet_dims)
        if 'aet_word_embeddings' in config:
            self.aet_word_embeddings.weight = nn.Parameter(torch.Tensor(config['aet_word_embeddings']))

        # if freeze, then don't need gradients
        if self.freeze_embs:
            self.word_embeddings.weight.requires_grad = False
            self.entity_embeddings.weight.requires_grad = False
            self.aet_entity_embeddings.weight.requires_grad = False
            if 'snd_word_voca' in config:
                self.snd_word_embeddings.weight.requires_grad = False

    def print_weight_norm(self):
        pass

    def save(self, path, suffix='', save_config=True):
        torch.save(self.state_dict(), path + '.state_dict' + suffix)

        if save_config:
            config = {'word_voca': self.word_voca.__dict__,
                      'entity_voca': self.entity_voca.__dict__}
            if 'snd_word_voca' in self.__dict__:
                config['snd_word_voca'] = self.snd_word_voca.__dict__
            if 'aet_word_voca' in self.__dict__:
                config['aet_word_voca'] = self.aet_word_voca.__dict__

            for k, v in self.__dict__.items():
                if not hasattr(v, '__dict__'):
                    config[k] = v

            with io.open(path + '.config', 'w', encoding='utf8') as f:
                json.dump(config, f)

    def load_params(self, path, param_names):
        params = torch.load(path)
        for pname in param_names:
            self._parameters[pname].data = params[pname]

    def loss(self, scores, grth):
        pass
