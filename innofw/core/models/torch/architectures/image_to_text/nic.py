import torch 

class CNNEncoder(torch.nn.Module):
    def __init__(self, 
                 backbone: torch.nn.Module, 
                 rnn_hidden_size: int,
        ):
        super().__init__()

        # According to Vinyals (2015), "it is natural to use a CNN as an image “encoder”, 
        # by first pre-training it for an image classification 
        # task and using the last hidden layer as an input to the RNN". 
        
        # Use only the last hidden layer as input to the RNN.
        self.feature_extractor = torch.nn.Sequential(*list(backbone.children())[:-1])

        # Remap the output of the CNN to the correct hidden size of the RNN.
        self.remap = torch.nn.LazyLinear(out_features=rnn_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms the input image to a feature vector.

        Args:
            x (torch.Tensor): Input image.

        Returns:
            torch.Tensor: Feature vector.

        Shape:
            - x: :math:`(N, C, H, W)`
            - Output: :math:`(N, rnn_hidden_size)`
        """
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        x = self.remap(x)
        return x
    

class RNNDecoder(torch.nn.Module):
    def __init__(
            self,
            rnn_hiden_size: int,
            vocab_size: int,
            max_sequence_length: int,
            start_token: int,
            end_token: int,
            pad_token: int,
        ) -> None:
        super().__init__()

        # As mentioned in Vinyals (2015), the image
        # and the words are mapped to the same space
        # using CNN encoder for images and Embedding
        # layer for words

        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=rnn_hiden_size,
        )

        self.rnn = torch.nn.LSTM(
            input_size=rnn_hiden_size,
            hidden_size=rnn_hiden_size,
            num_layers=1,
            batch_first=True,
        )

        self.fc = torch.nn.Linear(
            in_features=rnn_hiden_size,
            out_features=vocab_size,
        )
        self.vocabulary_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decodes an input feature vector of image to a sequence of words.

        Args:
            x (torch.Tensor): Input feature vector of image.
        
        Returns:
            torch.Tensor: Sequence of words.

        Shape:
            - x: :math:`(N, rnn_hidden_size)`
            - Output: :math:`(N, seq_len, vocab_size)`
        """
        length = 0
        batch_size = x.shape[0]

        # Indices of unfinished sequences
        indices = torch.full((batch_size,), True, dtype=torch.bool).to(x.device)
        
        # Output tensor (batch_size, seq_len, vocab_size)
        outputs = torch.zeros((batch_size, self.max_sequence_length, self.vocabulary_size)).to(x.device)

        # Set the first token to start token for all sequences
        outputs[:, :, self.start_token] = 1

        # Hidden and cell states (batch_size, rnn_hidden_size)
        hidden, cell = None, None

        while 1 + length < self.max_sequence_length and indices.any():
            # (batch_size, 1, rnn_hidden_size)
            if length == 0:
                step_input = x.unsqueeze(dim=1)
            else:
                step_input = self.embedding(outputs[indices, length].argmax(dim=-1, keepdim=True))

            if length == 0:
                step_output, (hidden, cell) = self.rnn(step_input)
            else:
                step_output, (hidden, cell) = self.rnn(step_input, (hidden, cell))
            outputs[indices, [1 + length]] = self.fc(step_output.squeeze(dim=1))

            indices = indices & (outputs[:, length].argmax(dim=-1) != self.end_token)
            length += 1
        
        return outputs
    

class NeuralImageCaption(torch.nn.Module):
    def __init__(
            self,
            backbone: torch.nn.Module,
            # preprocess: torch.nn.Module,
            rnn_hidden_size: int,
            vocab_size: int,
            max_sequence_length: int,
            start_token: int,
            end_token: int,
            pad_token: int,
        ) -> None:
        super().__init__()

        self.encoder = CNNEncoder(
            backbone=backbone,
            # preprocess=preprocess,
            rnn_hidden_size=rnn_hidden_size,
        )

        self.decoder = RNNDecoder(
            rnn_hiden_size=rnn_hidden_size,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            start_token=start_token,
            end_token=end_token,
            pad_token=pad_token,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms the input image to a sequence of words.

        Args:
            x (torch.Tensor): Input image.

        Returns:
            torch.Tensor: Sequence of words.

        Shape:
            - x: :math:`(N, C, H, W)`
            - Output: :math:`(N, seq_len, vocab_size)`
        """

        x = self.encoder(x)
        x = self.decoder(x)
        return x