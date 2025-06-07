# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

class MsgProcessor(torch.nn.Module):
    """
    Apply the secret message to the encoder output.
    Args:
        nbits: Number of bits used to generate the message. Must be non-zero
        hidden_size: Dimension of the message embedding
        msg_processor_type: Type of message processor. 
            First part indicates Gaussian or binary.
            Second part indicates the operation to apply to the latents.
    """
    def __init__(
        self, 
        nbits: int, 
        hidden_size: int,
        msg_processor_type: str = "binary+concat",
        msg_mult: float = 1.0,
    ):
        super().__init__()
        self.nbits = nbits
        self.hidden_size = hidden_size
        self.msg_mult = msg_mult
        # parse msg_processor_type
        self.msg_processor_type = msg_processor_type if nbits > 0 else "none+_"
        self.msg_type = self.msg_processor_type.split("+")[0]
        self.msg_agg = self.msg_processor_type.split("+")[1]
        # create msg embeddings
        if self.msg_type.startswith("no"):  # no message
            self.msg_embeddings = torch.tensor([])
        elif self.msg_type.startswith("bin"):  # binary
            self.msg_embeddings = torch.nn.Embedding(2 * nbits, hidden_size)
        elif self.msg_type.startswith("gau"):  # Gaussian
            self.msg_embeddings = torch.nn.Embedding(nbits, hidden_size)
        else:
            raise ValueError(f"Invalid msg_processor_type: {self.msg_processor_type}")

    def get_random_msg(self, bsz: int = 1, nb_repetitions = 1) -> torch.Tensor:
        """
        Generate a random message
        Args:
            bsz: Batch size
            nb_repetitions: Number of times to repeat the same message
        Returns:
            A random message tensor with shape BxL
        """
        if self.msg_type.startswith("bin"):
            if nb_repetitions != 1:
                assert self.nbits % nb_repetitions == 0, f"nbits must be divisible by nb_repetitions, got {self.nbits} and {nb_repetitions}"
                aux = torch.randint(0, 2, (bsz, self.nbits // nb_repetitions)) 
                return aux.unsqueeze(1).repeat(1, nb_repetitions, 1).view(bsz, self.nbits)
            else:
                return torch.randint(0, 2, (bsz, self.nbits))
        elif self.msg_type.startswith("gau"):
            gauss_vecs = torch.randn(bsz, self.nbits)
            gauss_vecs = gauss_vecs / torch.norm(gauss_vecs, dim=-1, keepdim=True)
            return gauss_vecs
        return torch.tensor([])

    def forward(
        self, 
        latents: torch.Tensor, 
        msg: torch.Tensor,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Apply the message to the latents.
        If no message: return latents as is
        If binary messages: create embeddings by selecting from the embedding layer, 
            then sum on the second dimension to get the message embeddings.
        If Gaussian messages: create embeddings by multiplying the message 
            with the weights of the embedding layer.

        Args:
            latents: The output of the encoder  Bx(d'xH/fxW/f)
            msg: The secret message to be embedded  BxL
        """

        if self.nbits == 0:
            return latents

        # create the message embeddings
        if self.msg_type.startswith("bin"):
            # create indices to take from embedding layer
            indices = 2 * torch.arange(msg.shape[-1]).to(msg.device)  # k: 0 2 4 ... 2k
            indices = indices.repeat(msg.shape[0], 1)  # b k
            indices = (indices + msg).long()
            # create embeddings
            msg_aux = self.msg_embeddings(indices)  # b k -> b k d
            msg_aux = msg_aux.sum(dim=-2)  # b k d -> b d
            msg_aux = msg_aux.unsqueeze(-1).unsqueeze(-1).repeat(
                1, 1, latents.shape[-2], latents.shape[-1]
            )  # b d -> b d h/f w/f
        elif self.msg_type.startswith("gau"):
            # create embeddings
            indices = torch.arange(msg.shape[-1]).to(msg.device).long()  # k: 0 1 2 ... k-1
            msg_aux = self.msg_embeddings(indices)  # k -> k d
            msg_aux = torch.einsum("kd, bk -> bd", msg_aux, msg)  # b d
            msg_aux = msg_aux.unsqueeze(-1).unsqueeze(-1).repeat(
                1, 1, latents.shape[-2], latents.shape[-1]
            )  # b d -> b d h/f w/f
        else:
            raise ValueError(f"Invalid msg_type: {self.msg_type}")

        # apply the message embeddings to the latents
        if self.msg_agg == "concat":
            latents = torch.cat([
                latents,  # b d' h/f w/f
                self.msg_mult * msg_aux  # b d h/f w/f
            ], dim=1)  # b d'+d h/f w/f
        elif self.msg_agg == "add":
            latents = latents + self.msg_mult * msg_aux  # -> b d' h/f w/f
        else:
            raise ValueError(f"Invalid msg_agg: {self.msg_agg}")
        
        if verbose:    
            print(f'indices: {indices.shape}')
            print(f'msgs: {msg.shape}')
            print(f'msg_aux: {msg_aux.shape}')
            print(f'latents: {latents.shape}')

        return latents
