import math

import torch

saved_photos = torch.load("activations.pt")

harmful_photos = saved_photos["ha"]
harmless_photos = saved_photos["hl"]

number_of_desks = harmful_photos.shape[1]

average_harmful_photo = harmful_photos.mean(0)
average_harmless_photo = harmless_photos.mean(0)

fingerprint_per_desk = average_harmful_photo - average_harmless_photo

fingerprint_strength = fingerprint_per_desk.norm(dim=-1, keepdim=True)
fingerprint_direction_per_desk = fingerprint_per_desk / fingerprint_strength

photo_scatter_per_desk = harmful_photos.std(0).mean(dim=-1)

separation_per_desk = fingerprint_strength.squeeze(-1) / photo_scatter_per_desk

print('desk  separation')

for desk in range(number_of_desks):
    score = separation_per_desk[desk].item()
    if not math.isfinite(score):
        # desk 0 is the embedding layer: the last token of the chat template is the
        # same for every prompt, so the spread and the gap are both exactly 0 -> 0/0
        print(f"{desk:4d}  {'n/a':>6s}  (constant across prompts)")
        continue
    print(f"{desk:4d}  {score:6.3f}  {'#' * int(score * 20)}")


ranked = torch.where(torch.isfinite(separation_per_desk),
                     separation_per_desk,
                     torch.full_like(separation_per_desk, float("-inf")))
print(f"\nbiggest gap at desk {ranked.argmax().item()}")
print("this is a hint for where to start testing -- not the answer")


# ---- save the 29 fingerprints for step 7 ----
torch.save({"dirs": fingerprint_direction_per_desk}, "directions.pt")