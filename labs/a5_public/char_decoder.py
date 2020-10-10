#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        super(CharDecoder, self).__init__()
        self.charDecoder = nn.LSTM(input_size=char_embedding_size,
                               hidden_size=hidden_size,
                               num_layers=1,
                               bias=True,
                               batch_first=False,
                               dropout=0,
                               bidirectional=False)
        self.char_output_projection = nn.Linear(hidden_size, len(target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(target_vocab.char2id), char_embedding_size, padding_idx=target_vocab.char2id['<pad>'])
        self.target_vocab = target_vocab
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        

        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        dec_embeddings = self.decoderCharEmb(input)
        H, dec_hidden = self.charDecoder(dec_embeddings, dec_hidden)
        scores = self.char_output_projection(H)
        return scores, dec_hidden
        ### TODO - Implement the forward pass of the character decoder. #?需要pack pad吗

        
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        input, groundtruth = char_sequence[:-1, :], char_sequence[1:, :]
        char_pad_mask = (input != self.target_vocab.char2id['<pad>'])  # (len, batch)
        scores, dec_hidden = self.forward(input, dec_hidden)  # (len, batch, vocab_size), 包括padding
        length, batch_size, vocab_size = scores.size()
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss_raw = criterion(scores.reshape(-1, vocab_size), groundtruth.reshape(-1))  # (len*b, )
        loss = torch.sum(loss_raw.view(length, batch_size) * char_pad_mask)
        return loss

        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).


        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        batch_size, hidden_size = initialStates[0].size(1), initialStates[0].size(2)
        vocab_size = len(self.target_vocab.char2id)
        decodedWords, output_words = [], []
        output_chars = torch.zeros(batch_size, max_length, device=device, dtype=torch.long)
        initialChars = torch.ones(1, batch_size, device=device, dtype=torch.long) * self.target_vocab.start_of_word
        chars, states = initialChars, initialStates
        for i in range(max_length):
            scores, states = self.forward(chars, states)
            chars = scores.argmax(-1)
            output_chars[:, i] = scores.view(batch_size, vocab_size).argmax(dim=-1)
        for word in output_chars:
            end_indices = (word == self.target_vocab.end_of_word).nonzero()
            if len(end_indices) == 0:
                output_words.append(''.join([self.target_vocab.id2char[c] for c in word.tolist()]))
            else:
                output_words.append(''.join([self.target_vocab.id2char[c] for c in word[:end_indices[0].item()].tolist()]))
        return output_words

        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.


        ### END YOUR CODE
