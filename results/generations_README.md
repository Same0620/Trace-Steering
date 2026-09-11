# generations.jsonl

Generation steers every position except absolute position 0, INCLUDING generated tokens (steer.Steer.all_but_first). Base rows use the same path at alpha=0.

- model: Qwen/Qwen3-8B, steer layer 17/36
- openers: 20 (items/openers.txt)
- arms per organism: base (alpha=0) + [('mu_D', 1.0), ('mu_D', 2.0), ('mu_Dprime_matched', 1.0)]
- sampling: temperature 0.7, top_p 0.95, max_new_tokens 60
- seed: zlib.crc32(f'{opener_idx}|{arm}|{alpha}'.encode()) (seed_check for (0, 'mu_D', 1.0) = 291864001); recorded per row
- every sample saved; no filtering
- base samples identical across organisms: True
- rows: 160
