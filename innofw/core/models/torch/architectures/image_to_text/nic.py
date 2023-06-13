import torch

from innofw.core.models.torch.architectures.image_to_text.base import ImageToText 

class CNNEncoder(torch.nn.Module):
    """This module produces a feature vector from an input image.
    The feature vector is then used as input to the RNN decoder.

    The reference paper:
    
    Vinyals, Oriol, et al. "Show and tell: A neural image caption generator."
    Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
    """


    def __init__(self, 
                 backbone: torch.nn.Module, 
                 rnn_hidden_size: int,
        ):
        """Initializes the module.

        Args:
            backbone (torch.nn.Module): CNN backbone.
            rnn_hidden_size (int): Hidden size of the RNN.
        """
        super().__init__()

        # According to Vinyals (2015), "it is natural to use a CNN as an image “encoder”, 
        # by first pre-training it for an image classification 
        # task and using the last hidden layer as an input to the RNN". 
        
        # Use only the last hidden layer as input to the RNN.
        self.feature_extractor = torch.nn.Sequential(*list(backbone.children())[:-1])

        # Freeze the weights of the CNN backbone.
        for module in [*self.feature_extractor.children()][-5:]: # enable the last 5 layers to be trained
            module.requires_grad_ = True

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
        x = self.feature_extractor(x)  # Extract features from the image
        x = torch.flatten(x, start_dim=1)  # Flatten the output of the CNN to a vector
        x = self.remap(x)  # Change the size of the vector to the correct hidden size of the RNN
        return x
    

class RNNDecoder(torch.nn.Module):
    """This module decodes an input feature vector of image to a sequence of words.

    The reference paper:

    Vinyals, Oriol, et al. "Show and tell: A neural image caption generator."
    Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
    """

    def __init__(
            self,
            rnn_hiden_size: int,
            vocab_size: int,
            max_sequence_length: int,
            start_token: int,
            end_token: int,
            pad_token: int,
        ) -> None:
        """Initializes the module.

        Args:
            rnn_hiden_size (int): Hidden size of the RNN.
            vocab_size (int): Size of the vocabulary.
            max_sequence_length (int): Maximum length of the sequence.
            start_token (int): Start token.
            end_token (int): End token.
            pad_token (int): Pad token.
        """
        super().__init__()

        # As mentioned in Vinyals (2015), the image
        # and the words are mapped to the same space
        # using CNN encoder for images and Embedding
        # layer for words

        # Embedding layer (vocab_size, rnn_hidden_size)
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=rnn_hiden_size,
        )

        # RNN (rnn_hidden_size, rnn_hidden_size)
        self.rnn = torch.nn.LSTM(
            input_size=rnn_hiden_size,
            hidden_size=rnn_hiden_size,
            num_layers=1,
            batch_first=True,
        )

        # Fully connected layer (rnn_hidden_size -> vocab_size)
        self.fc = torch.nn.Linear(
            in_features=rnn_hiden_size,
            out_features=vocab_size,
        )

        # Save the parameters
        self.vocabulary_size = vocab_size 
        self.max_sequence_length = max_sequence_length
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
    

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, forcing=False) -> torch.Tensor:
        """Decodes an input feature vector of image to a sequence of words.

        Args:
            x (torch.Tensor): Input feature vector of image.
            y (torch.Tensor, optional): Sequence of words to be used as input to the RNN. Defaults to None.
            forcing (bool, optional): Whether to use teacher forcing. Defaults to False.
        
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
                # Use the feature vector of the image as the first input
                step_input = x.unsqueeze(dim=1) 
            else: # else and if not merged to keep the idea clear
                if forcing:
                    # Use the ground truth as the input to the RNN
                    step_input = self.embedding(y[indices, length].unsqueeze(dim=1))
                else:
                    # Use the output of the RNN as the input to the RNN
                    step_input = self.embedding(outputs[indices, length].argmax(dim=-1, keepdim=True))
    
            if length == 0:
                # Use all inputs on the first step
                step_output, (hidden, cell) = self.rnn(step_input)
            else:
                # Use only inputs of unfinished sequences
                step_output, (hidden[:, indices], cell[:, indices]) = self.rnn(
                    step_input, (hidden[:, indices], cell[:, indices])
                )
            
            # Compute the output of the RNN
            outputs[indices, [1 + length]] = self.fc(step_output.squeeze(dim=1))
            # Update the indices of unfinished sequences
            indices = indices & (outputs[:, length].argmax(dim=-1) != self.end_token)
            length += 1
        
        return outputs
    

class NeuralImageCaption(ImageToText):
    """This module implements the image captioning model.

    The reference paper:

    Vinyals, Oriol, et al. "Show and tell: A neural image caption generator."
    Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

    The module assumes a tokenizer is used and uses the size of the vocabulary
    to determine the size of the embedding layer.
    """


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
        """Initializes the module.

        Args:
            backbone (torch.nn.Module): Backbone CNN.
            preprocess (torch.nn.Module): Preprocess module.
            rnn_hidden_size (int): Hidden size of the RNN.
            vocab_size (int): Size of the vocabulary (for embeddings).
            max_sequence_length (int): Maximum length of the sequence.
        """

        super().__init__()

        self.encoder = CNNEncoder(
            backbone=backbone,
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

    def forward(self, image: torch.Tensor, captions: torch.Tensor=None, forcing=False) -> torch.Tensor:
        """Transforms the input image to a sequence of words.

        Args:
            x (torch.Tensor): Input image.

        Returns:
            torch.Tensor: Sequence of words.

        Shape:
            - x: :math:`(N, C, H, W)`
            - Output: :math:`(N, seq_len, vocab_size)`
        """

        x = self.encoder(image)
        x = self.decoder(x, y=captions, forcing=forcing)
        return x