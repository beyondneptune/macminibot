# Todo

Bootstrap backlog for `infrastructure`. Each item is a self-contained agent-sized epic.

- [ ] **Pick language stack** — Decide hot-path vs. research-path stack (e.g. Python everywhere, or Python + Rust hot-path). Update `CONTEXT.md` and `shared-core/CONTEXT.md` with the choice.
- [ ] **Repo bootstrap** — Initialize `git`, `package.json` / `pyproject.toml` / `Cargo.toml` per language choice. Add `README.md`, `.editorconfig`, `.gitignore`.
- [ ] **Lint + format + typecheck** — Configure linters and formatters (ruff/mypy for Python, clippy/rustfmt for Rust). Pre-commit hook + CI enforcement.
- [ ] **Test scaffolding** — Pick test runner per language, add one passing smoke test, wire to CI.
- [ ] **CI pipeline** — Set up CI (GitHub Actions or chosen alternative): typecheck, lint, test on every push. Caching for deps.
- [ ] **Secrets backend** — Pick approach (macOS Keychain via `security` CLI, sops + age, vault). Build a single helper API used by every dept. No API keys in env vars in plaintext shell rc.
- [ ] **launchd supervision** — `.plist` files for each long-running process, checked into repo with deploy script. KeepAlive on crash with backoff.
- [ ] **Time sync monitor** — Ensure chrony/macOS NTP is running; expose clock-drift metric to `observability`; alert on drift > threshold.
- [ ] **Audit log primitive** — Append-only signed log used by `risk-management`, `execution`, and `control-plane` for sensitive events.
- [ ] **Backup strategy** — Decide and implement backups for tick archive, model registry, trade journal. Document restore drill.
