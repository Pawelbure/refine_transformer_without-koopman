# Branch workflow notes

This repository now contains an `improved_transformer` branch that carries the transformer stability updates. You can keep a dedicated working directory for this branch alongside your main checkout using Git worktrees:

1. From your main clone directory, create a sibling folder for the branch checkout:
   ```bash
   git worktree add ../improved_transformer-worktree improved_transformer
   ```
   This will populate `../improved_transformer-worktree` with the contents of the `improved_transformer` branch.

2. To enter the branch-specific working tree and continue working on that branch:
   ```bash
   cd ../improved_transformer-worktree
   ```

3. When you want to remove the extra directory later:
   ```bash
   git worktree remove ../improved_transformer-worktree
   ```

This keeps branch checkouts separated without requiring multiple clones while sharing the same `.git` metadata.
