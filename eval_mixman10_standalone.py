import argparse, json, os
from typing import List

import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import clip_nopooling as clip
from clip_nopooling.clip import get_word_id

class CocoVal(Dataset):
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        with open(os.path.join(self.root, "val_annotation.json")) as f:
            self.img_list = json.load(f)
        with open(os.path.join(self.root, "category.json")) as f:
            cat2idx = json.load(f)
        self.num_classes = len(cat2idx)
        self.labels = [""] * self.num_classes
        for name, idx in cat2idx.items():
            self.labels[idx] = name
        self.preprocess = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index: int):
        item = self.img_list[index]
        image = self.preprocess(Image.open(os.path.join(self.root, "val2014", item["file_name"])).convert("RGB"))
        target = torch.full((self.num_classes,), -1, dtype=torch.int)
        target[item["labels"]] = 1
        return image, target


def compute_map(scores: Tensor, targets: Tensor) -> float:
    aps = []
    for k in range(scores.size(1)):
        s, t = scores[:, k], targets[:, k]
        _, idx = torch.sort(s, descending=True)
        pos = 0.0
        total = 0.0
        acc = 0.0
        for i in idx:
            lbl = t[i].item()
            if lbl == 1:
                pos += 1
            total += 1
            if lbl == 1:
                acc += pos / total
        aps.append(acc / pos if pos != 0 else 0.0)
    return float(sum(aps) / len(aps))


class PositionEmbeddingLearned(nn.Module):
    def __init__(self, num_pos_feats: int):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        xe = self.col_embed(torch.arange(w, device=x.device))
        ye = self.row_embed(torch.arange(h, device=x.device))
        return torch.cat([
            xe.unsqueeze(0).repeat(h, 1, 1),
            ye.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)


class SpatialFeatureAdapter(nn.Module):
    def __init__(self, dim_feat: int, dim_space: int, alpha: float):
        super().__init__()
        self.layer1 = nn.Conv2d(dim_feat, dim_space, 1)
        self.layer2 = nn.Conv2d(dim_space, dim_feat, 1)
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        y = self.layer2(F.relu(self.layer1(x)))
        return self.alpha * y + (1.0 - self.alpha) * x


class PositionalEncoder(nn.Module):
    def __init__(self, dim_feat: int, num_layer: int):
        super().__init__()
        self.position_embedding = PositionEmbeddingLearned(dim_feat // 2)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim_feat, 8, batch_first=True, activation=F.leaky_relu),
            num_layer,
            nn.LayerNorm(dim_feat, eps=1e-5),
        )

    def forward(self, x: Tensor) -> Tensor:
        bs, c, h, w = x.size()
        x = x + self.position_embedding(x)
        x = self.encoder(x.flatten(2, 3).permute(0, 2, 1))
        return x.permute(0, 2, 1).reshape(bs, c, h, w)


class Baseline(nn.Module):
    def __init__(self, labels: List[str]):
        super().__init__()
        self.dim_feat = 512
        self.num_labels = len(labels)
        self.len_adj = 16
        self.len_prompt = 77
        self.feature_resolution = 14

        model, _ = clip.load("RN101", device="cpu")
        self.image_encoder = model.visual
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.logit_scale = model.logit_scale
        self.text_projection = model.text_projection
        self.backbone = model

        self.labels_ids_in_clip = clip.tokenize(labels)
        self.norm = nn.BatchNorm2d(self.dim_feat)
        self.adj_embedding = nn.Embedding(self.num_labels, self.dim_feat * self.len_adj)
        self.image_adapter = SpatialFeatureAdapter(2048, 1024, 0.5)
        self.image_conv = nn.Conv2d(2048, self.dim_feat, 1)
        self.caption_encoder14 = PositionalEncoder(self.dim_feat, 2)
        self.caption_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(self.dim_feat, 8, batch_first=True, activation=F.leaky_relu),
            2,
            nn.LayerNorm(self.dim_feat, eps=1e-5),
        )
        self.pos_mask = nn.Embedding(1, self.dim_feat)
        self.caption_fc = nn.Linear(self.dim_feat, self.num_labels, bias=False)

    def _build_prompt_bank(self, device) -> Tensor:
        clip_emb = [self.token_embedding(torch.tensor(ids, device=device)) for ids in self.labels_ids_in_clip]
        self_emb = self.adj_embedding(torch.arange(self.num_labels, device=device)).reshape(
            self.num_labels, self.len_adj, self.dim_feat
        )
        info_slots = torch.zeros(self.num_labels, dtype=torch.long, device=device)
        prompts = torch.zeros((self.num_labels, self.len_prompt, self.dim_feat), device=device)
        sot = self.token_embedding(torch.tensor(get_word_id("<|" + "startoftext|>"), device=device))
        eot = self.token_embedding(torch.tensor(get_word_id("<|" + "endoftext|>"), device=device))
        prompts[:, 0] = sot
        for k, (adj, word) in enumerate(zip(self_emb, clip_emb)):
            t = torch.cat([adj, word], dim=0)
            assert 1 + t.size(0) + 1 <= 77
            prompts[k, 1:1 + t.size(0)] = t
            prompts[k, 1 + t.size(0)] = eot
            info_slots[k] = 1 + t.size(0)
        x = prompts + self.positional_embedding
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.ln_final(x)
        return x[torch.arange(x.shape[0]), info_slots] @ self.text_projection

    def forward(self, image: Tensor) -> Tensor:
        bs = image.size(0)
        if not hasattr(self, "prompt_bank"):
            self.prompt_bank = self._build_prompt_bank(image.device)
        pb = self.prompt_bank

        feat = self.image_encoder(image)
        if self.feature_resolution != 14:
            feat = F.interpolate(feat, size=(self.feature_resolution, self.feature_resolution))
        feat = self.image_conv(self.image_adapter(feat))

        text_norm = pb / pb.norm(dim=1, keepdim=True)
        img_flat = feat.flatten(2, 3).permute(0, 2, 1).contiguous()
        img_norm = img_flat / img_flat.norm(dim=2, keepdim=True)
        grid_logits = (self.logit_scale.exp() * img_norm @ text_norm.t()).mean(dim=1)

        t = self.norm(self.caption_encoder14(feat))
        memory = (t + feat).flatten(2, 3).permute(0, 2, 1).contiguous()
        query = pb.unsqueeze(0).repeat(bs, 1, 1)
        ans = self.caption_decoder(memory=memory, tgt=query)
        logits = self.caption_fc(ans)
        n = logits.size(1)
        return logits[:, torch.arange(n), torch.arange(n)]


def load_ema(model: Baseline, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "custom" not in ckpt or "model_ema" not in ckpt["custom"]:
        raise KeyError("checkpoint has no custom.model_ema")
    prefix = "ema_model."
    state = {k[len(prefix):]: v for k, v in ckpt["custom"]["model_ema"].items() if k.startswith(prefix)}
    if not state:
        raise KeyError("no 'ema_model.' keys in model_ema state")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", required=True)
    args = p.parse_args()

    dev = torch.device("cuda")
    valset = CocoVal(args.data_root)
    model = Baseline(valset.labels)
    load_ema(model, args.ckpt)
    model.eval().to(dev)

    loader = DataLoader(valset, batch_size=32, shuffle=False,
                        num_workers=8, pin_memory=True)

    all_scores, all_targets = [], []
    with torch.inference_mode():
        for image, target in loader:
            logits = model(image.to(dev, non_blocking=True))
            all_scores.append(logits.float().cpu())
            all_targets.append(target.long())
    scores = torch.cat(all_scores, dim=0)
    targets = torch.cat(all_targets, dim=0)
    print(f"mAP: {compute_map(scores, targets):.6f}")


if __name__ == "__main__":
    main()
