import torch
from typing import Dict, Union

class BasicImageToTextEncoderDecoder(torch.nn.Module):
    class Encoder(torch.nn.Module):
        def __init__(self,
                     encoder_network_size: int,
                     dropout_probability: float,
                     backbone_model: torch.nn.Module,
                     image_transforms: torch.nn.Module,
                     average_pool_size: int,
                     fine_tune_backbone: bool
                     ):
            super().__init__()
            self.network_size = encoder_network_size
            self.image_encoder = torch.nn.Sequential(*backbone_model.children())[:-2]
            self.transforms = image_transforms
            self.avg_pool = torch.nn.AdaptiveAvgPool2d(average_pool_size)
            self.linear = torch.nn.LazyLinear(encoder_network_size)
            
            self.relu = torch.nn.ReLU()
            self.dropout = torch.nn.Dropout(dropout_probability)

            for param in self.image_encoder.parameters():
                param.requires_grad = False

            for module in [*self.image_encoder.children()][:-5]:
                for param in module.parameters():
                    param.requires_grad = fine_tune_backbone
            
        
        def forward(self, image: torch.Tensor) -> torch.Tensor:     
            X = self.transforms(image)
            X = self.image_encoder(X)
            X = self.avg_pool(X) 
            X = X.permute(0, 2, 3, 1)
            X = self.linear(X)
            X = self.dropout(self.relu(X))
            return X.flatten(start_dim=1, end_dim=-2)

    def __init__(self,
                decoder_hidden_size: int,
                embedding_size: int,
                dropout_probability: float,
                # lstm_size: int,
                encoder_network_size: int,
                encoder_backbone_model: torch.nn.Module,
                encoder_transforms: torch.nn.Module,
                encoder_avg_pool_size: int,
                encoder_fine_tune: bool,
                word2int: Dict[str, int]=None,
            ):

            super().__init__()
            self.decoder_hidden_size = decoder_hidden_size
            self.embedding_size = embedding_size
            self._initialization_complete = False

            if word2int is not None:   
                self.initialize(word2int)

            self.encoder = BasicImageToTextEncoderDecoder.Encoder(
                encoder_network_size=encoder_network_size, 
                dropout_probability=dropout_probability,
                backbone_model=encoder_backbone_model,
                image_transforms=encoder_transforms,
                fine_tune_backbone=encoder_fine_tune,
                average_pool_size=encoder_avg_pool_size,
            )

           
            self.lstm = torch.nn.LSTMCell(
                input_size=encoder_avg_pool_size ** 2 * encoder_network_size * decoder_hidden_size,
                hidden_size=decoder_hidden_size
            )

            self.device = torch.device('cpu')

            self.f_beta = torch.nn.Linear(encoder_network_size, decoder_hidden_size)
            self.dropout = torch.nn.Dropout(dropout_probability)

            self.initial_hidden = torch.nn.Linear(encoder_network_size, decoder_hidden_size)
            self.initial_cell = torch.nn.Linear(encoder_network_size, decoder_hidden_size)


    @property
    def is_ready(self):
        return self._initialization_complete


    def initialize(self, word2int):
        self.vocabulary_size = len(word2int)
        self.START = word2int["<s>"]

        self.embedding = torch.nn.Embedding(
            self.vocabulary_size, self.embedding_size
        )
        self.linear = torch.nn.Linear(self.decoder_hidden_size, self.vocabulary_size)
        self._initialization_complete = True
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)


    def forward(self, 
                images: torch.Tensor,
                captions: Union[torch.Tensor, None],
                max_caption_length: int,
                teacher_forcing=False):

        assert self._initialization_complete, \
            "Trying to use uninitialized model. Please, run `.initialize(word2int)` \
            before you call other method of the model."
        
        if captions is None and teacher_forcing:
            raise ValueError("Running with `teacher_forcing=True`, but not \
                                providing target captions.")
        
        encoded_images = self.encoder(images)
        batch_size = encoded_images.size(0)

        h = torch.zeros(batch_size, self.decoder_hidden_size).type_as(images).to(torch.float)
        c = torch.zeros(batch_size, self.decoder_hidden_size).type_as(images).to(torch.float)

        predictions = torch.zeros(
            batch_size,
            max_caption_length, 
            self.vocabulary_size
        ).type_as(images).to(torch.float)

        if teacher_forcing: embeddings = self.embedding(captions)
        else: embeddings = self.embedding(torch.full((batch_size, 1,), self.START).type_as(images))
        for t in range(max_caption_length):
            lstm_i = embeddings[:, [t], :] + encoded_images.flatten(start_dim=1).unsqueeze(2)
            lstm_i = lstm_i.flatten(start_dim=1)
            
            h, c = self.lstm(
               lstm_i,
                (h, c)
            )
        
            predictions[:, t, :] = self.linear(self.dropout(h))
            if not teacher_forcing:
                embeddings = torch.cat([embeddings, self.embedding(predictions[:, [t], :].argmax(dim=-1))], dim=1)
        return predictions