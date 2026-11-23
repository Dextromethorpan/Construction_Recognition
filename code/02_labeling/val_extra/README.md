# val_extra labels (Claude only, post-test confirmation set V3_confirm)
labels.txt: one line per image index K (order in labels/relations/v2x/val_extra_order.txt):
  K | ID SUP HEL ; ... ; add x,y,w,h SUP HEL ; ...
  ID = proposal index (proposals.json); X = delete (not a person / duplicate)
  SUP: O on scaffold (platform, frame, attached ladder) | G not on scaffold (ground, deck, other surface) | U unclear
  HEL: W worn | N not worn | H held | V not visible
  'K | skip' = no usable people
Unlisted proposals are treated as deleted.
