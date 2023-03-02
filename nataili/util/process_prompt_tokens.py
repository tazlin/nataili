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
import requests

from nataili.util import load_learned_embed_in_clip, load_learned_embed_in_clip_v2


def process_prompt_tokens(prompt_tokens, model, concepts_dir, model_baseline):
    # compviz codebase
    # tokenizer =  model.cond_stage_model.tokenizer
    # text_encoder = model.cond_stage_model.transformer
    # diffusers codebase
    # tokenizer = pipe.tokenizer
    # text_encoder = pipe.text_encoder

    ext = (".pt", ".bin")
    remote_embeds_file = "https://raw.githubusercontent.com/Sygil-Dev/nataili-model-reference/main/db_embeds.json"
    available_embeds = []

    # Fetch the embeds db
    r = requests.get(remote_embeds_file)
    embeds = r.json()
    for embed in embeds:
        available_embeds.append(embed)

    for token_name in prompt_tokens:
        embedding_path = os.path.join(concepts_dir, token_name)
        if available_embeds[token_name]["baseline"] == model_baseline:
            for files in os.listdir(embedding_path):
                if files.endswith(ext):
                    load_learned_embed_in_clip(
                        f"{os.path.join(embedding_path, files)}",
                        model.cond_stage_model.transformer,
                        model.cond_stage_model.tokenizer,
                        f"<{token_name}>",
                    )
        else:
            print(f"Concept {token_name} not found in {concepts_dir}")
            return
