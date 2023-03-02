"""
This file is part of nataili ("Homepage" = "https://github.com/Sygil-Dev/nataili").

Copyright 2022 hlky and Sygil-Dev
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import os

import torch


def load_learned_embed_in_clip_v2(learned_embeds_path, model, text_encoder, tokenizer, token=None):
    print (f"Token = {token}")
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cuda:0")
    # separate token and the embeds
    if learned_embeds_path.endswith(".pt"):
        # old format
        # token = * so replace with file directory name when converting
        trained_token = os.path.basename(learned_embeds_path)
        print (f"Trained token = {trained_token}")
        params_dict = {trained_token: torch.tensor(list(loaded_learned_embeds["string_to_param"].items())[0][1])}
        print (f"Params Dict = {params_dict}")
        learned_embeds_path = os.path.splitext(learned_embeds_path)[0] + ".bin"
        torch.save(params_dict, learned_embeds_path)
        loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cuda:0")
        trained_token = list(loaded_learned_embeds.keys())[0]
        print (f"Trained token (post torch.load) = {trained_token}")
        embeds = loaded_learned_embeds[trained_token]
        print (f"Embeds = {embeds}")
    elif learned_embeds_path.endswith(".bin"):
        trained_token = list(loaded_learned_embeds.keys())[0]
        print (f"Trained token (post torch.load) = {trained_token}")
        embeds = loaded_learned_embeds[trained_token]
        print (f"Embeds = {embeds}")

    embeds = loaded_learned_embeds[trained_token]
    # cast to dtype of text_encoder
    dtype = text_encoder.get_cast_dtype()
    embeds.to(dtype)
    print (f"Embeds (post send to dtype) = {embeds}")

    model.cond_stage_model.model.token_embedding.weight = torch.nn.Parameter(torch.cat((model.cond_stage_model.model.token_embedding.weight, embeds),dim=0))
    print (f"Return Token = {token}")
    return token