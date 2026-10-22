---
name: debug-loop
description: Reproduce-first bug fixing loop — localize with the code graph, confirm a failing reproduction, apply a minimal patch, verify with tests, repair at most twice.
---

# Debug loop

1. `search_similar_code` with the issue's key terms; open the top hits with `read_file`.
2. `get_code_subgraph` around the most suspicious function to see callers/callees before editing.
3. Write the reproduction as a small file and run it with `run_command`; record the exact failure line.
4. Edit with `edit_file` (smallest possible hunk).
5. Re-run reproduction + the module's relevant tests with `run_command`.
6. On failure: one hypothesis per repair round, max 2 rounds, then `submit_patch` with the best attempt so far.
